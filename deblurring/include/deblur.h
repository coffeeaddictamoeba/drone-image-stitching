#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <unordered_map>

class Deblurrer {
    public:
        void createTestBlurredImage();
        void createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF);
        void applySyntheticBlur(const std::string &inputPath, const std::string &outputImagePath, bool grayscale);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float snr);
    
        float calculateGSD(const std::string& imagePath);
    
    private:
        void fftShift(cv::Mat& input);
        cv::Mat createSyntheticTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createSyntheticMetadata();
    };