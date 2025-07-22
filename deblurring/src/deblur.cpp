#include "../include/deblur.h"
#include "../include/metadata.h"
#include <cstring>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <unordered_map>

cv::Mat Deblurrer::createSyntheticTestImage(int width = 640, int height = 480) {
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::line(canvas, cv::Point(50, 100), cv::Point(600, 100), cv::Scalar(0, 0, 0), 2);
    cv::line(canvas, cv::Point(50, 200), cv::Point(600, 200), cv::Scalar(0, 0, 0), 3);
    cv::line(canvas, cv::Point(100, 400), cv::Point(500, 100), cv::Scalar(0, 0, 255), 2);
    cv::line(canvas, cv::Point(300, 450), cv::Point(600, 250), cv::Scalar(255, 0, 0), 2);

    cv::circle(canvas, cv::Point(320, 240), 40, cv::Scalar(0, 255, 0), -1); // filled

    cv::putText(canvas, "Test Image", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    return canvas;
}

std::unordered_map<std::string, std::string> Deblurrer::createSyntheticMetadata() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> altitudeDist(50.0, 150.0);       // meters
    std::uniform_real_distribution<> focalLengthDist(2.0, 5.0);       // mm
    std::uniform_real_distribution<> speedDist(3.0, 15.0);            // m/s
    std::uniform_real_distribution<> yawDist(0.0, 360.0);             // degrees
    std::uniform_real_distribution<> exposureDist(1.0 / 500.0, 1.0 / 10.0); // seconds

    float altitude = altitudeDist(gen);
    float focalLength = focalLengthDist(gen);
    float speed = speedDist(gen);
    float yaw = yawDist(gen);
    float exposure = exposureDist(gen);

    std::unordered_map<std::string, std::string> metadata = {
        { "GPS Altitude", std::to_string(altitude) },
        { "Focal Length", std::to_string(focalLength) },
        { "FlightSpeed", std::to_string(speed) },
        { "DroneYaw", std::to_string(yaw) },
        { "Exposure Time", std::to_string(exposure) }
    };

    std::cout << "Randomized metadata:\n";
    for (const auto& kv : metadata) {
        std::cout << "  " << kv.first << " = " << kv.second << "\n";
    }

    return metadata;
}

void Deblurrer::createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat &syntheticPSF) {
    int ksize = std::max(blurLengthPx * 2 + 1, 9);  // ensure visible blur
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

    std::unordered_map<std::string, std::string> metadata = extractImageMetadata(inputPath);
    float speed = std::stof(metadata["FlightSpeed"]);
    float yaw = std::stof(metadata["DroneYaw"]);
    float exposure = std::stof(metadata["Exposure Time"]);

    float gsd = calculateGSD(inputPath); // mm/px
    float blur = speed * 1000.0f * exposure; // mm
    int blurLength = static_cast<int>(blur / gsd); // px
    blurLength = std::max(1, blurLength);

    std::cout << "Original metadata: speed = " << speed << " m/s, yaw = " << yaw << " deg, exposure = " << exposure << " s (" << blurLength << " px blur)" << std::endl;    

    cv::Mat psf;
    createSyntheticPSF(blurLength, yaw, psf);
    std::cout << "PSF size: " << psf.cols << "x" << psf.rows << std::endl;

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    imwrite(outputImagePath, blurred);

    std::unordered_map<std::string, std::string> customTags;
    customTags["FlightSpeed"] = std::to_string(speed);
    customTags["DroneYaw"] = std::to_string(yaw);
    copyMetadata(inputPath, outputImagePath, customTags);

    std::cout << "Blurred image saved to: " << outputImagePath << "\n";
}

void Deblurrer::createTestBlurredImage() {
    std::string imageSynthetic = "synthetic.jpg";
    std::string blurredImage = "synthetic_blurred.jpg";

    cv::Mat synthetic = createSyntheticTestImage();
    cv::imwrite(imageSynthetic, synthetic);

    std::unordered_map<std::string, std::string> metadata = createSyntheticMetadata();
    assignMetadata(imageSynthetic, metadata);

    applySyntheticBlur(imageSynthetic, blurredImage, false);
}

float Deblurrer::calculateGSD(const std::string& imagePath) {
    std::unordered_map<std::string, std::string> metadata = extractImageMetadata(imagePath);

    float alt = std::stof(metadata["GPS Altitude"]); // m
    float flen = std::stof(metadata["Focal Length"]); // mm
    float sensorWidth = 3.68f; // mm
    float sensorHeight = 2.76f; // mm

    cv::Mat img = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    int imageWidth = img.cols; // px
    int imageHeight = img.rows; // px

    alt *= 1000.0f; // m -> mm

    float gsdWidth = (alt * sensorWidth) / (flen * imageWidth); // mm/px
    float gsdHeight = (alt * sensorHeight) / (flen * imageHeight); // mm/px

    float gsd = std::max(gsdWidth, gsdHeight); // mm/px

    std::cout << "GSD: Calculated GSD = " << gsd<< " mm/px\n";
    std::cout << "  Focal Length: " << flen << " mm\n";
    std::cout << "  Sensor Size: " << sensorWidth << "x" << sensorHeight << " mm\n";
    std::cout << "  Image Resolution: " << imageWidth << "x" << imageHeight << " px\n";
    std::cout << "  Altitude: " << alt << " mm\n";

    return gsd;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--generate-test") == 0) { // create synthetic image
        Deblurrer db;
        db.createTestBlurredImage();
        return 0;
    } else if (argc == 3 and std::strcmp(argv[1], "-b") == 0) { // blur image instead of deblurring (useful for testing)
        std::string imageToBlur = argv[2];

        std::string imageName = imageToBlur.substr(0, imageToBlur.find('.'));
        std::string imageExtension = imageToBlur.substr(imageToBlur.find('.'), imageToBlur.size());
        std::string blurredImage = imageName + "_blurred" + imageExtension;

        Deblurrer db;
        db.applySyntheticBlur(imageToBlur, blurredImage, false);
        return 0;
    } else if (argc == 2) { // deblurring logic
        std::string blurredImage = argv[1];

        std::string imageName = blurredImage.substr(0, blurredImage.find('.'));
        std::string imageExtension = blurredImage.substr(blurredImage.find('.'), blurredImage.size());
        std::string deblurredImage = imageName + "_deblurred" + imageExtension;

        Deblurrer deblurrer;

        std::cout << "Deblurring complete.\n";
        return 0;
    } else {
        std::cerr << "Wrong arguments.\n";
        return 1;
    }
}

