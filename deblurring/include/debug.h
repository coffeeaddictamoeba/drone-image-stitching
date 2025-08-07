#ifndef DEBLUR_DEBUG
#define DEBLUR_DEBUG

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

inline void visualizeMatrix(const cv::Mat &image, const std::string &outputImagePath) {
    if (image.empty()) {
        std::cerr << "[ERROR] visualizeMatrix: Failed to load image." << std::endl;
        return;
    }

    cv::Mat debug;
    cv::normalize(image, debug, 0, 255, cv::NORM_MINMAX);
    debug.convertTo(debug, CV_8U);
    cv::imwrite(outputImagePath, debug);
}

inline void visualizeMagnitude(const cv::Mat &complexImage, const std::string &outputImagePath) {
    if (complexImage.empty()) {
        std::cerr << "[ERROR] visualizeMagnitude: Failed to load image." << std::endl;
        return;
    }

    std::vector<cv::Mat> planes;
    cv::split(complexImage, planes);

    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    mag += 1e-5f;

    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    visualizeMatrix(mag, outputImagePath);
}

inline void diffBlurredDeblurred(cv::Mat &blurred, cv::Mat &deblurred, const std::string &outputImagePath) {
    if (blurred.empty()) {
        std::cerr << "[ERROR] diffBlurredDeblurred: Failed to load blurred image." << std::endl;
        return;
    } else if (deblurred.empty()) {
        std::cerr << "[ERROR] diffBlurredDeblurred: Failed to load deblurred image." << std::endl;
        return;
    }

    if (blurred.size() != deblurred.size()) {
        std::cerr << "[ERROR] diffBlurredDeblurred: Size mismatch.\n";
        return;
    }

    if (deblurred.type() != blurred.type()) {
        deblurred.convertTo(deblurred, blurred.type());
    }

    cv::Mat diff = cv::abs(output - blurred);
    visualizeMatrix(diff, outputImagePath);
}

inline void countMatrixZeros(const cv::Mat &image, const std::string &imageName) {
    int zeros = 0;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j << image.cols; j++) {
            if (image.at<float>(i, j) < 1e-4f) {
                zeros++;
                std::cout << "[" << imageName << "] Zero at (" << i << ", " << j << ").\n";
            }
        }
    }
    std::cout << "Total zeros in "<< imageName << " : " << zeros << std::endl;
}

#endif