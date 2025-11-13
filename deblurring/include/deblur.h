#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <opencv2/opencv.hpp>

#include "config.h"

class Deblurrer {
    public:
        Deblurrer() = default;
        Deblurrer(DeblurConfig &config);
        ~Deblurrer() = default;

        void generateTest(const fs::path &output = "");
        void blurImage(const fs::path &imgpath, bool grayscale);
        void deblurImage(const fs::path &imgpath, float snr);
        bool isBlurred(const fs::path &imgpath, float blurThreshold);
        void saveImage(const cv::Mat &image, const fs::path &input, const std::string &prefix);
    
    private:
        DeblurConfig config_;

        void fftshift(cv::Mat& input);
        void estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat& syntheticPSF);
        void wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& output, float blurLength, float snr);
        void denoiseImage(cv::Mat& image, float strength, float edgeStrength);
        bool isBlurred(const cv::Mat& image, float blurThreshold, int maxImageSize);

        void findBlurLength(const fs::path& imgpath, int& blurLength, float& blurAngleRad, std::unordered_map<std::string, std::string>& md);
        void findVBodies(const std::unordered_map<std::string, std::string>& md, float& Vx, float& Vy, float& Vz, float& speed);

        void saveImage(const cv::Mat& image, const fs::path& input, const std::unordered_map<std::string, std::string>& md, const std::string& prefix);

        cv::Mat psfdft(const cv::Mat& normPSF, cv::Size targetSize);
        cv::Mat padInput(const cv::Mat& input, const cv::Mat& psf);
        cv::Mat createSNRMap(cv::Size size, const cv::Mat& psf);
        cv::Mat buildWienerDenominator(const cv::Mat& psfMag2, const cv::Mat& snrMap, float snr, const cv::Mat& inputF, const cv::Mat& psf);
        std::pair<cv::Mat, cv::Mat> psfConjMag(const cv::Mat& psfDFT);
        std::array<cv::Mat, 3> splitInputChannels(const cv::Mat& inputF);
        std::array<cv::Mat, 3> deconvolve(const std::array<cv::Mat, 3>& inputChannels, const cv::Mat& psfConj, const cv::Mat& wienerDenom, const cv::Mat& hann, const cv::Mat& originalInput, const cv::Mat& psf);
        
        inline const cv::Size getDFTSize(cv::Size s);

        cv::Mat createTestImage(int width, int height);
        std::unordered_map<std::string, std::string> createTestMetadata();
    };