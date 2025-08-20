#ifndef METADATA_H
#define METADATA_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// info
void listMetadata();

// metadata operations
std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath);
void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath, const std::unordered_map<std::string, std::string>& customTags = {});
void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags);
std::string extractExifTagValue(const std::string& imagePath, const std::string& tagName);

// parsing operations
float parseExifExposureTime(const std::string &exposure_str);
float parseExifGPSSpeed(const std::string &gpsspeed_str, const std::string &gpsspeedref_str);
#endif