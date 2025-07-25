#include "config.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv4/opencv2/opencv.hpp>
#include <unordered_map>

class Deblurrer {
    public:
        Deblurrer() = default;
        Deblurrer(DeblurConfig &config);
        ~Deblurrer() = default;

        void generateTest(const std::string &testOutputPath = "");
        void blurImage(const std::string &inputPath, const std::string &outputImagePath, bool grayscale);
        void deblurImage(const std::string &inputPath, const std::string &outputImagePath, float snr);
    
    private:
        // for arguments handling
        DeblurConfig config_;

        // for actual deblurring
        float calculateGSD(const std::string& imagePath);
        void fftShift(cv::Mat& input);
        void estimatePSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float snr);
        void denoiseImage(cv::Mat& image, float strength, float edgeStrength);
        void recoverBrightness(cv::Mat& image, float gamma);

        // for testing purposes
        cv::Mat createTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createTestMetadata();
    };