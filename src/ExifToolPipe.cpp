#include "../include/metadata.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sstream>

ExifToolPipe::ExifToolPipe() {
    int pipeToChild[2], pipeFromChild[2];
    pipe(pipeToChild);
    pipe(pipeFromChild);

    childPid = fork();
    if (childPid == 0) {
        dup2(pipeToChild[0], STDIN_FILENO);
        dup2(pipeFromChild[1], STDOUT_FILENO);
        close(pipeToChild[1]);
        close(pipeFromChild[0]);
        execlp("exiftool", "exiftool", "-stay_open", "True", "-@", "-", nullptr);
        exit(1);
    }

    writeFd = pipeToChild[1];
    readFd = pipeFromChild[0];
    close(pipeToChild[0]);
    close(pipeFromChild[1]);
}

ExifToolPipe::~ExifToolPipe() {
    std::string quit = "-stay_open\nFalse\n";
    write(writeFd, quit.c_str(), quit.size());
    close(writeFd);
    close(readFd);
    waitpid(childPid, nullptr, 0);
}

bool ExifToolPipe::sendCommand(const std::string& imagePath) {
    std::ostringstream cmd;
    cmd << imagePath << "\n-n\n-execute\n";
    std::string commandStr = cmd.str();
    return write(writeFd, commandStr.c_str(), commandStr.size()) >= 0;
}

std::string ExifToolPipe::readResponse() {
    std::string result;
    char buffer[256];
    ssize_t bytes;
    while ((bytes = read(readFd, buffer, sizeof(buffer) - 1)) > 0) {
        buffer[bytes] = 0;
        result += buffer;
        if (result.find("{ready}") != std::string::npos)
            break;
    }
    return result;
}

std::map<std::string, std::string> ExifToolPipe::getLastExifData() {
    std::string response = readResponse();
    std::map<std::string, std::string> data;
    std::istringstream stream(response);
    std::string line;
    while (std::getline(stream, line)) {
        if (line.find("{ready}") != std::string::npos) break;
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            data[key] = val;
        }
    }
    return data;
}