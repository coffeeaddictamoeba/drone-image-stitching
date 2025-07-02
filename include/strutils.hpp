#pragma once

#include <string>
#include <string_view>

namespace utils {

    inline int to_int(std::string_view s) {
        int value = 0, sign = 1;
        size_t i = 0;
        if (!s.empty() && s[0] == '-') {
            sign = -1;
            i = 1;
        }
        for (; i < s.size(); ++i) value = value * 10 + (s[i] - '0');
        return sign * value;
    }

    inline double to_double(std::string_view s) {
        double value = 0.0;
        int sign = 1;
        size_t i = 0;
    
        if (!s.empty() && s[0] == '-') {
            sign = -1;
            i = 1;
        }
    
        for (; i < s.size() && s[i] != '.'; ++i) {
            value = value * 10.0 + (s[i] - '0');
        }
    
        if (i < s.size() && s[i] == '.') {
            ++i;
            for (double frac = 0.1; i < s.size(); frac *= 0.1, ++i) {
                value += (s[i] - '0') * frac;
            }
        }
        return sign * value;
    }

    inline std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r\f\v");
        if (first == std::string::npos) return str;
        size_t last = str.find_last_not_of(" \t\n\r\f\v");
        return str.substr(first, last - first + 1);
    }
}