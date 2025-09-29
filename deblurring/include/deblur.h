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
        void blurImage(const std::string &inputImagePath, bool grayscale);
        void deblurImage(const std::string &inputImagePath, float snr);
        bool isBlurred(const std::string &imagePath, float blurThreshold);
        void saveImage(const cv::Mat &image, const std::string &inputImagePath, const std::string &prefix);
    
    private:
        // For arguments handling
        DeblurConfig config_;

        // For actual deblurring
        float findBlurLength(const std::string &imagePath, float &blurAngleRad);
        float calculateGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth, float sensorHeight);
        void fftShift(cv::UMat& input);
        void estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat &syntheticPSF);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float blurLength, float snr);
        void denoiseImage(cv::Mat& image, float strength, float edgeStrength);
        bool isBlurred(const cv::Mat &image, float blurThreshold);

        // Metadata reading helpers
        void findPitchRollYawFromMetadata(std::unordered_map<std::string, std::string> &metadata, float &pitchRad, float &rollRad, float &yawRad);
        void findSpeedFromMetadata(std::unordered_map<std::string, std::string> &metadata, float &speedX, float &speedY, float &speedZ);
        void findVBodies(std::unordered_map<std::string, std::string> &metadata, float &Vx, float &Vy, float &Vz, float &speed);
        float findGSDFromMetadata(std::unordered_map<std::string, std::string> &metadata);
        float findGPSImgDirectionFromMetadata(std::unordered_map<std::string, std::string> &metadata);

        // Deblur helpers
        cv::UMat padInput(const cv::Mat& input, const cv::Mat& psf);
        cv::Mat normalizePSF(const cv::Mat& psf);
        cv::UMat psfdft(const cv::Mat& normPSF, cv::Size targetSize);
        std::pair<cv::UMat, cv::UMat> psfConjMag(const cv::UMat& psfDFT);
        cv::Mat createSNRMap(cv::Size size, const cv::Mat& psf);
        cv::Mat buildWienerDenominator(const cv::Mat& psfMag2, const cv::Mat& snrMap, float snr, const cv::Mat& inputF, const cv::Mat& psf);
        std::vector<cv::Mat> splitInputChannels(const cv::Mat& inputF);
        cv::Mat createFeatherMask(cv::Size size);
        std::vector<cv::Mat> deconvolve(const std::vector<cv::Mat>& inputChannels, const cv::Mat& psfConj, const cv::Mat& wienerDenom, const cv::Mat& hann, const cv::Mat& mask, const cv::Mat& originalInput, const cv::Mat& psf);

        // For testing purposes
        cv::Mat createTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createTestMetadata();
    };