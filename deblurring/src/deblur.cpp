#include "../include/deblur.h"
#include <cstring>
#include <opencv4/opencv2/opencv.hpp>
#include <string>

void Deblurrer::createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF) {
    int ksize = blurLengthPx * 2 + 1;
    syntheticPSF = cv::Mat::zeros(cv::Size(ksize, ksize), CV_32F);
    cv::Point center(ksize / 2, ksize / 2);
    cv::Point pt2(center.x + int(blurLengthPx * cos(yaw * CV_PI / 180.0)),
              center.y + int(blurLengthPx * sin(yaw * CV_PI / 180.0)));
    cv::line(syntheticPSF, center, pt2, cv::Scalar::all(1), 1, cv::LINE_AA);
    cv::Scalar sumVal = sum(syntheticPSF);
    syntheticPSF /= sumVal[0];
}

void Deblurrer::applySyntheticBlur(const std::string &inputPath, const std::string &outputImagePath, bool grayscale) {
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat normal = cv::imread(inputPath, imreadFlag);
    if (normal.empty()) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return; 
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> speedDist(2.0, 10.0); // m/s
    std::uniform_real_distribution<> yawDist(0.0, 360.0); // deg
    std::uniform_real_distribution<> exposureDist(1.0 / 200.0, 1.0 / 30.0); // s

    float speed = speedDist(gen);
    float yaw = yawDist(gen);
    float exposure = exposureDist(gen);

    float gsd_mm_per_px = 1.0;
    float blur_mm = speed * 1000.0f * exposure;
    int blur_len_px = static_cast<int>(blur_mm / gsd_mm_per_px);
    blur_len_px = std::max(1, blur_len_px);

    std::cout << "Synthetic metadata: speed = " << speed << " m/s, yaw = " << yaw
         << " deg, exposure = " << exposure << " s (" << blur_len_px << " px blur)" << std::endl;

    cv::Mat psf;
    createSyntheticPSF(blur_len_px, yaw, psf);

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    imwrite(outputImagePath, blurred);

    std::ostringstream oss;
    oss << "exiftool -overwrite_original "
        << "-UserComment=" << "FlightSpeed=" << speed << ",DroneYaw=" << yaw << " "
        << "-ExposureTime=" << exposure << " "
        << outputImagePath;

    std::string embedCmd = oss.str();
    system(embedCmd.c_str());

    std::cout << "Blurred image saved to: " << outputImagePath << "\n";
}

int main(int argc, char** argv) {
    if (argc == 2) {
        // logic for deblurring
        return 0;
    } else if (argc == 3 and std::strcmp(argv[1], "-b") == 0) { // blur image instead of deblurring (useful for testing)
        std::string imageToBlur = argv[2];

        std::string imageName = imageToBlur.substr(0, imageToBlur.find('.'));
        std::string imageExtension = imageToBlur.substr(imageToBlur.find('.'), imageToBlur.size());
        std::string blurredImage = imageName + "_blurred" + imageExtension;

        Deblurrer db;
        db.applySyntheticBlur(imageToBlur, blurredImage, false);
        return 0;
    } else {
        std::cerr << "Wrong arguments.\n";
        return 1;
    }
}

