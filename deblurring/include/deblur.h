#include "config.h"
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv2/opencv.hpp>
#include <unordered_map>

class Deblurrer {
    public:
        Deblurrer() = default;
        Deblurrer(DeblurConfig &config);
        ~Deblurrer() = default;

        void generateTest(const std::string &testOutputPath = "");
        void blurImage(const std::string &inputImagePath, const std::string &outputImagePath, bool grayscale);
        void deblurImage(const std::string &inputImagePath, const std::string &outputImagePath, float snr);
        bool isBlurred(const std::string &imagePath, float blurThreshold);
    
    private:
        // for arguments handling
        DeblurConfig config_;

        // for actual deblurring
        float findBlurLength(const std::string &imagePath, float &blurAngleRad);
        float calculateGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth, float sensorHeight);
        void fftShift(cv::Mat& input);
        void estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat &syntheticPSF);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float snr);
        void denoiseImage(cv::Mat& image, float strength, float edgeStrength);
        void recoverBrightness(cv::Mat& image, float gamma);
        bool isBlurred(const cv::Mat &image, float blurThreshold);

        // for testing purposes
        cv::Mat createTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createTestMetadata();
    };