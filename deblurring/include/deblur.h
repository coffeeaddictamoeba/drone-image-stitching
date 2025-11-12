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

        void generateTest(const fs::path &testOutputPath = "");
        void blurImage(const fs::path &inputImagePath, bool grayscale);
        void deblurImage(const fs::path &inputImagePath, float snr);
        bool isBlurred(const fs::path &imagePath, float blurThreshold);
        void saveImage(const cv::Mat &image, const fs::path &inputImagePath, const std::string &prefix);
    
    private:
        // For arguments handling
        DeblurConfig config_;

        // For actual deblurring
        void fftshift(cv::Mat& input);
        void estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat &syntheticPSF);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float blurLength, float snr);
        void denoiseImage(cv::Mat& image, float strength, float edgeStrength);
        bool isBlurred(const cv::Mat &image, float blurThreshold, int maxImageSize);

        // Metadata reading helpers
        void findBlurLength(const fs::path &imagePath, int &blurLength, float &blurAngleRad);
        void findVBodies(const std::unordered_map<std::string, std::string> &metadata, float &Vx, float &Vy, float &Vz, float &speed);

        // Deblur helpers
        cv::Mat padInput(const cv::Mat& input, const cv::Mat& psf);
        cv::Mat psfdft(const cv::Mat& normPSF, cv::Size targetSize);
        std::pair<cv::Mat, cv::Mat> psfConjMag(const cv::Mat& psfDFT);
        cv::Mat createSNRMap(cv::Size size, const cv::Mat& psf);
        cv::Mat buildWienerDenominator(const cv::Mat& psfMag2, const cv::Mat& snrMap, float snr, const cv::Mat& inputF, const cv::Mat& psf);
        std::array<cv::Mat, 3> splitInputChannels(const cv::Mat& inputF);
        cv::Mat createFeatherMask(cv::Size size);
        std::array<cv::Mat, 3> deconvolve(const std::array<cv::Mat, 3>& inputChannels, const cv::Mat& psfConj, const cv::Mat& wienerDenom, const cv::Mat& hann, const cv::Mat& originalInput, const cv::Mat& psf);
        inline const cv::Size getDFTSize(cv::Size s);

        // For testing purposes
        cv::Mat createTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createTestMetadata();
    };