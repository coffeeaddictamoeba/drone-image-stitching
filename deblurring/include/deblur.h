#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <unordered_map>

class Deblurrer {
    public:
        void createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF);
        void applySyntheticBlur(const std::string &inputPath, const std::string &outputImagePath, bool grayscale);
    
        float calculateGSD(const std::string& imagePath);
    
    private:
        // Helper function for FFT-based deconvolution
        void calculateDft(const cv::Mat& input, cv::Mat& output);
        void convolveWithDft(const cv::Mat& imageDFT, const cv::Mat& psfDFT, cv::Mat& outputDFT);
        void circularShift(cv::Mat& img, int dx, int dy);
    };