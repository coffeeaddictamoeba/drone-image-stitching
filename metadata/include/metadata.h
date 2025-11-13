#pragma once

#ifndef METADATA_H
#define METADATA_H

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <unordered_map>
#include <unistd.h>
#include <cstdio>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdexcept>

namespace metadata {
    // NOTE: this list is a metadata minimum required for deblurring/image stitching
    constexpr const std::array<std::string_view, 23> EXIFTOOL_TAGS = {
        "CameraModelName", 
        "Make", 
        "Software", 
        "ModifyDate", 
        "ExposureTime", 
        "ISO",
        "DateTimeOriginal", 
        "CreateDate", 
        "FocalLength",
        "GPSVersionID", 
        "GPSLatitude", 
        "GPSLongitude", 
        "GPSAltitude",
        "GPSImgDirection", 
        "GPSImgDirectionRef", 
        "GPSSpeed", 
        "GPSSpeedRef",
        "FlightPitchDegree", 
        "FlightYawDegree", 
        "FlightRollDegree",
        "XMP-drone-dji:FlightXSpeed", 
        "XMP-drone-dji:FlightYSpeed", 
        "XMP-drone-dji:FlightZSpeed"
    };

    const std::unordered_map<std::string, std::string> XMP_TAGS_MAP = {
        {"FlightXSpeed", "XMP-drone-dji:FlightXSpeed"},
        {"FlightYSpeed", "XMP-drone-dji:FlightYSpeed"},
        {"FlightZSpeed", "XMP-drone-dji:FlightZSpeed"}
    };

    inline constexpr std::size_t TAGS_ARGS_LEN = []() consteval {
        std::size_t sum = 0;
        for (auto tag : EXIFTOOL_TAGS) sum += 2 + tag.size(); // " -" + tag
        return sum;
    }();

    template <std::size_t M>
    consteval auto tagsToArgs() {
        std::array<char, M + 1> out{}; // +1 for '\0'
        std::size_t pos = 0;
        for (auto tag : EXIFTOOL_TAGS) {
            out[pos++] = ' ';
            out[pos++] = '-';
            for (char c : tag) out[pos++] = c;
        }
        out[pos] = '\0';
        return out;
    }

    inline constexpr auto TAG_ARGS_BUF = tagsToArgs<TAGS_ARGS_LEN>();
    inline constexpr std::string_view EXIFTOOL_TAGS_ARGS(
        TAG_ARGS_BUF.data(),
        TAG_ARGS_BUF.size() - 1 // drop '\0'
    );

    #endif

    class ExifToolSession {
    public:
        ExifToolSession() {
            #ifdef _WIN32
                proc_ = _popen("exiftool -n -q -q -fast2", "r");
                stayOpen_ = false; // no stay_open for a while
            #else
                int inpipe[2]{}, outpipe[2]{};
                if (pipe(inpipe) < 0 || pipe(outpipe) < 0) {
                    throw std::runtime_error("pipe() failed for ExifToolSession");
                }

                pid_ = fork();
                if (pid_ < 0) {
                    throw std::runtime_error("fork() failed for ExifToolSession");
                }

                // child process: exiftool
                if (pid_ == 0) {
                    // stdin <- inpipe[0], stdout/stderr -> outpipe[1]
                    dup2(inpipe[0], STDIN_FILENO);
                    dup2(outpipe[1], STDOUT_FILENO);
                    dup2(outpipe[1], STDERR_FILENO);

                    close(inpipe[0]);
                    close(inpipe[1]);
                    close(outpipe[0]);
                    close(outpipe[1]);

                    constexpr const char* argv[] = {
                        "exiftool",
                        "-stay_open", "True",
                        "-@", "-",              // argfile from stdin
                        "-common_args",
                        "-n", "-q", "-q", "-fast2",
                        nullptr
                    };
                    execvp(argv[0], const_cast<char* const*>(argv));
                    _exit(127); // if exec fails
                }

                // parent process
                close(inpipe[0]);   // only write to child's stdin
                close(outpipe[1]);  // only read from child's stdout

                in_fd_  = inpipe[1];
                out_fd_ = outpipe[0];
                stayOpen_ = true;
            #endif
        }

        ~ExifToolSession() {
            #ifdef _WIN32
                if (proc_) {
                    _pclose(proc_);
                    proc_ = nullptr;
                }
            #else
                if (stayOpen_ && in_fd_ >= 0) {
                    constexpr char term[] = "-stay_open\nFalse\n-execute\n";
                    writeAll(in_fd_, term, sizeof(term) - 1);
                }
                if (in_fd_  >= 0) { close(in_fd_);  in_fd_  = -1; }
                if (out_fd_ >= 0) { close(out_fd_); out_fd_ = -1; }
                if (pid_ > 0) {
                    int status = 0;
                    waitpid(pid_, &status, 0); // reap child
                }
            #endif
        }

        std::string run(const std::string& args) {
            #ifdef _WIN32
                return runOneShot(args);
            #else
                if (!stayOpen_) return runOneShot(args);

                std::array<std::string, 64> tokens;
                tokenizeArgs(args, tokens);
                if (tokens.empty()) return {};

                for (auto& t : tokens) {
                    writeAll(in_fd_, t.data(), t.size());
                    writeAll(in_fd_, "\n", 1);
                }

                constexpr char trailer[] = "-echo3\n{ready}\n-execute\n";
                writeAll(in_fd_, trailer, sizeof(trailer) - 1);

                return readUntilReady();
            #endif
        }

    private:
        #ifdef _WIN32
            FILE* proc_ = nullptr;
            bool  stayOpen_ = false;

            std::string runOneShot(const std::string& args) {
                std::string cmd = "exiftool -n -q -q -fast2 " + args;
                FILE* p = _popen(cmd.c_str(), "r");
                if (!p) throw std::runtime_error("Failed to run exiftool one-shot (Windows).");
                std::string out;
                char buf[1024];
                while (fgets(buf, sizeof(buf), p)) out += buf;
                _pclose(p);
                return out;
            }
        #else
            pid_t pid_    = -1;
            int   in_fd_  = -1;        // write end -> child's stdin
            int   out_fd_ = -1;        // read end  <- child's stdout
            bool  stayOpen_ = false;

            void writeAll(int fd, const char* p, size_t n) {
                while (n) {
                    ssize_t w = ::write(fd, p, n);
                    if (w < 0) {
                        if (errno == EINTR) continue;
                        throw std::runtime_error("write() to exiftool failed");
                    }
                    n -= (size_t)w;
                    p += w;
                }
            }

            // handle spaces, quotes, backslashes
            void tokenizeArgs(const std::string& line, std::array<std::string, 64>& out) {
                int i = 0;
                std::string cur; 
                cur.reserve(64);
                bool inS = false, inD = false, esc = false;

                for (char c : line) {
                    if (esc) { 
                        cur.push_back(c);
                        esc = false;
                        continue;
                    }

                    if (c == '\\') { esc = true; continue; }
                    else if (c == '\'' && !inD) { inS = !inS; continue; }
                    else if (c == '"'  && !inS) { inD = !inD; continue; }

                    if (!inS && !inD && std::isspace((unsigned char)c)) {
                        if (!cur.empty()) { out[i++] = std::move(cur); cur.clear(); cur.reserve(64); }
                        continue;
                    }
                    cur.push_back(c);
                }
                if (!cur.empty()) out[i++] = std::move(cur);
            }

            std::string readUntilReady() {
                static constexpr char kReady[] = "{ready}\n";
                std::string out; out.reserve(4096);
                char buf[8192];
                for (;;) {
                    ssize_t r = ::read(out_fd_, buf, sizeof(buf));
                    if (r > 0) {
                        out.append(buf, buf + r);
                        if (out.size() >= sizeof(kReady) - 1) {
                            auto pos = out.rfind(kReady);
                            if (pos != std::string::npos) {
                                out.erase(pos); // drop marker
                                break;
                            }
                        }
                        continue;
                    }
                    if (r == 0) break; // EOF (unexpected with stay_open True)
                    if (errno == EINTR) continue;
                    throw std::runtime_error("read() from exiftool failed");
                }
                return out;
            }

            std::string runOneShot(const std::string& args) {
                std::string cmd = "exiftool -n -q -q -fast2 " + args + " 2>/dev/null";
                FILE* p = popen(cmd.c_str(), "r");
                if (!p) throw std::runtime_error("Failed to run exiftool one-shot.");
                std::string out; char buf[1024];
                while (fgets(buf, sizeof(buf), p)) out += buf;
                pclose(p);
                return out;
            }
        #endif
    };

    bool isExifToolAvailable();
    bool isValidTag(std::string_view tag);

    void listMetadata();

    std::string extract(const std::string& imgpath, const std::string& tagname);
    std::unordered_map<std::string, std::string> extractAll(const std::string& imgpath);

    //void copy(/* kv pair? */ const std::string& imgpath); // single tag copy
    void copyAll(const std::string& srcpath, const std::string& dstpath);
    void copyAll(const std::unordered_map<std::string, std::string>& md, const std::string& imgpath);

    void getPitchRollYaw(const std::unordered_map<std::string, std::string>& md, float &pitch, float &roll, float &yaw);
    void getPitchRollYawRad(const std::unordered_map<std::string, std::string> &md, float &pitchRad, float &rollRad, float &yawRad);
    void getSpeedXYZ(const std::unordered_map<std::string, std::string> &md, float &speedX, float &speedY, float &speedZ);

    float getGPSImgDirection(const std::unordered_map<std::string, std::string> &md);
    float getGPSImgDirectionRad(const std::unordered_map<std::string, std::string> &md);

    float findGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth, float sensorHeight);
    float findGSD(const std::unordered_map<std::string, std::string>& md, float sensorWidth, float sensorHeight);

    bool tagAsFloat(const std::unordered_map<std::string,std::string>& md, const std::string& key, float& value);
    bool tagAsInt(const std::unordered_map<std::string,std::string>& md, const std::string& key, int& value);
}
