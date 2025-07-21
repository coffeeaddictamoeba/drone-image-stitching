#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <opencv4/opencv2/opencv.hpp>

class Deblurrer {
    public:
        void applySyntheticBlur(const std::string &inputPath, const std::string &outputImagePath, bool grayscale);
    private:
        void createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF);
};