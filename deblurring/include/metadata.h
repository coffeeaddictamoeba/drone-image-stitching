#pragma once

#ifndef METADATA_H
#define METADATA_H

#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <array>
#include <unordered_map>
#include <vector>

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

class ExifToolSession {
    public:
        ExifToolSession() {
            proc_ = nullptr;
            #ifdef _WIN32
                proc_ = _popen("exiftool -stay_open True -@ -", "w+");
            #else
                proc_ = popen("exiftool -stay_open True -@ -", "r+");
                if (!proc_) {
                    proc_ = popen("exiftool -stay_open True -@ -", "w+");
                }
            #endif
    
            if (proc_) {
                stayOpen_ = true;
            } else {
                stayOpen_ = false;
            }
        }
    
        ~ExifToolSession() {
            if (stayOpen_ && proc_) {
                fprintf(proc_, "-stay_open\nFalse\n-execute\n");
                fflush(proc_);
                char buf[1024];
                while (fgets(buf, sizeof(buf), proc_)) { /* ignore */ }
                pclose(proc_);
            }
        }
    
        std::string run(const std::string &args) {
            if (stayOpen_) {
                return runStayOpen(args);
            } else {
                return runOneShot(args);
            }
        }
    
    private:
        FILE* proc_ = nullptr;
        bool stayOpen_ = false;
    
        std::string runStayOpen(const std::string &args) {
            if (!proc_) throw std::runtime_error("ExifTool -stay_open not available.");
    
            fprintf(proc_, "%s\n-execute\n", args.c_str());
            fflush(proc_);
    
            std::string output;
            char buf[1024];
            while (fgets(buf, sizeof(buf), proc_)) {
                if (strstr(buf, "{ready}")) {
                    break;
                }
                output += buf;
            }
            return output;
        }
    
        std::string runOneShot(const std::string &args) {
            std::string one = std::string("exiftool ") + args;

            auto pipeDeleter = [](FILE* fp) noexcept {
                if (fp) pclose(fp);
            };
            
            std::unique_ptr<FILE, decltype(pipeDeleter)> pipe(popen(one.c_str(), "r"), pipeDeleter);            
            if (!pipe) throw std::runtime_error("Failed to run exiftool one-shot command.");
    
            std::string out;
            char buf[1024];
            while (fgets(buf, sizeof(buf), pipe.get())) {
                out += buf;
            }
            return out;
        }
};

// info
void listMetadata();

// metadata operations
std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath);
void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath);
void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags);
std::string extractExifTagValue(const std::string& imagePath, const std::string& tagName);

// parsing operations
float parseExifExposureTime(const std::string &exposure_str);
float parseExifGPSSpeed(const std::string &gpsspeed_str, const std::string &gpsspeedref_str);
#endif