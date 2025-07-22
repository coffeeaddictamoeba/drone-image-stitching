#ifndef METADATA_H
#define METADATA_H

#include <string>
#include <unordered_map>
#include <vector>

std::unordered_map<std::string, std::string> extractImageMetadata(const std::string& imagePath);
void copyMetadata(const std::string& sourceImagePath, const std::string& destImagePath, const std::unordered_map<std::string, std::string>& customTags = {});
void assignMetadata(const std::string& imagePath, const std::unordered_map<std::string, std::string>& tags);

#endif