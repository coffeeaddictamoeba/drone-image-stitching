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

void Deblurrer::createSyntheticPSF(int blurLengthPx, float yaw, cv::Mat& syntheticPSF) {
    int ksize = std::max(blurLengthPx * 2 + 1, 15);

    syntheticPSF = cv::Mat::zeros(ksize, ksize, CV_32F);

    float angleRad = yaw * CV_PI / 180.0f;

    cv::Point center(ksize / 2, ksize / 2);
    cv::Point pt1(center.x - std::round(blurLengthPx * 0.5f * std::cos(angleRad)),
                  center.y - std::round(blurLengthPx * 0.5f * std::sin(angleRad)));
    cv::Point pt2(center.x + std::round(blurLengthPx * 0.5f * std::cos(angleRad)),
                  center.y + std::round(blurLengthPx * 0.5f * std::sin(angleRad)));

    cv::line(syntheticPSF, pt1, pt2, cv::Scalar(1.0f), 1, cv::LINE_AA);

    double sumVal = cv::sum(syntheticPSF)[0];
    if (sumVal == 0.0) {
        std::cerr << "[ERROR] PSF line generation failed — blurLengthPx: " << blurLengthPx << ", yaw: " << yaw << "\n";
        return;
    }

    syntheticPSF /= static_cast<float>(sumVal);

    cv::Mat psfDebug;
    cv::normalize(syntheticPSF, psfDebug, 0, 255, cv::NORM_MINMAX);
    psfDebug.convertTo(psfDebug, CV_8U);
    cv::imwrite("debug_psf.png", psfDebug);
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

void Deblurrer::fftShift(cv::Mat& input) {
    input = input(cv::Rect(0, 0, input.cols & -2, input.rows & -2));  // make even size
    int cx = input.cols / 2;
    int cy = input.rows / 2;

    cv::Mat q0(input, cv::Rect(0, 0, cx, cy));   // top-left
    cv::Mat q1(input, cv::Rect(cx, 0, cx, cy));  // top-right
    cv::Mat q2(input, cv::Rect(0, cy, cx, cy));  // bottom-left
    cv::Mat q3(input, cv::Rect(cx, cy, cx, cy)); // bottom-right

    cv::Mat tmp;
    q0.copyTo(tmp);  q3.copyTo(q0);  tmp.copyTo(q3);
    q1.copyTo(tmp);  q2.copyTo(q1);  tmp.copyTo(q2);
}

void Deblurrer::wienerDeconvolution(const cv::Mat& blurred, const cv::Mat& psf, cv::Mat& outputImage, float snr = 300.0) {
    if (blurred.empty() || psf.empty()) {
        std::cerr << "[Error] Input image or PSF is empty.\n";
        outputImage = cv::Mat();
        return;
    }

    cv::Mat psf32f;
    psf.convertTo(psf32f, CV_32F);
    double psfSum = cv::sum(psf32f)[0];
    if (psfSum <= 0.0) {
        std::cerr << "[Error] PSF sum is zero or negative.\n";
        outputImage = cv::Mat::zeros(blurred.size(), blurred.type());
        return;
    }
    psf32f /= psfSum;

    std::vector<cv::Mat> channels;
    cv::Mat result;

    if (blurred.channels() == 3) {
        cv::split(blurred, channels);
    } else {
        channels.push_back(blurred.clone());
    }

    std::vector<cv::Mat> deblurredChannels;

    for (size_t i = 0; i < channels.size(); ++i) {
        cv::Mat channelFloat;
        channels[i].convertTo(channelFloat, CV_32F, 1.0 / 255.0);
        cv::normalize(channelFloat, channelFloat, 0, 1, cv::NORM_MINMAX);

        // Pad and shift PSF
        cv::Mat paddedPSF = cv::Mat::zeros(channelFloat.size(), CV_32F);
        int x = (paddedPSF.cols - psf32f.cols) / 2;
        int y = (paddedPSF.rows - psf32f.rows) / 2;
        psf32f.copyTo(paddedPSF(cv::Rect(x, y, psf32f.cols, psf32f.rows)));
        fftShift(paddedPSF);

        // DFTs
        cv::Mat channelDFT, psfDFT;
        cv::dft(channelFloat, channelDFT, cv::DFT_COMPLEX_OUTPUT);
        cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);

        // PSF magnitude squared
        std::vector<cv::Mat> psfPlanes(2);
        cv::split(psfDFT, psfPlanes);
        cv::Mat psfMag2;
        cv::magnitude(psfPlanes[0], psfPlanes[1], psfMag2);
        psfMag2 = psfMag2.mul(psfMag2);

        // Conjugate of PSF
        psfPlanes[1] = -psfPlanes[1];
        cv::Mat psfConj;
        cv::merge(psfPlanes, psfConj);

        // Denominator
        float K = 1.0f / snr;
        cv::Mat denom = psfMag2 + K;

        // Numerator: blurred * conj(psf)
        std::vector<cv::Mat> blurredPlanes(2);
        cv::split(channelDFT, blurredPlanes);
        cv::Mat numReal = blurredPlanes[0].mul(psfPlanes[0]) - blurredPlanes[1].mul(psfPlanes[1]);
        cv::Mat numImag = blurredPlanes[0].mul(psfPlanes[1]) + blurredPlanes[1].mul(psfPlanes[0]);

        // Element-wise division
        cv::Mat realPart = numReal / denom;
        cv::Mat imagPart = numImag / denom;

        // Inverse DFT
        std::vector<cv::Mat> wienerPlanes{realPart, imagPart};
        cv::Mat wienerDFT;
        cv::merge(wienerPlanes, wienerDFT);

        cv::Mat deconvolved;
        cv::idft(wienerDFT, deconvolved, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        // Normalize channel
        double minVal, maxVal;
        cv::minMaxLoc(deconvolved, &minVal, &maxVal);

        if (minVal == maxVal || std::isnan(minVal) || std::isnan(maxVal)) {
            std::cerr << "[Warning] Channel " << i << " is flat or invalid.\n";
            deconvolved = cv::Mat::zeros(deconvolved.size(), CV_32F);
        } else {
            deconvolved = (deconvolved - minVal) / (maxVal - minVal);
        }

        cv::Mat output8U;
        deconvolved.convertTo(output8U, CV_8U, 255);
        deblurredChannels.push_back(output8U);
    }

    if (deblurredChannels.size() == 1) {
        outputImage = deblurredChannels[0];
    } else {
        cv::merge(deblurredChannels, outputImage);
    }

    cv::imwrite("debug_deblurred_rgb.png", outputImage);
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

        cv::Mat blurred = cv::imread(blurredImage);
        if (blurred.empty()) {
            std::cerr << "Failed to load image: " << blurredImage << "\n";
            return 1;
        }

        auto metadata = extractImageMetadata(blurredImage);
        float speed = std::stof(metadata["FlightSpeed"]);
        float yaw = std::stof(metadata["DroneYaw"]);
        float exposure = std::stof(metadata["Exposure Time"]);
        float gsd = deblurrer.calculateGSD(blurredImage);
        float blur_mm = speed * 1000.0f * exposure;
        int blur_px = std::max(1, static_cast<int>(blur_mm / gsd));

        std::cout << "Deblurring with blur length: " << blur_px << " px, yaw: " << yaw << " deg\n";

        cv::Mat psf;
        deblurrer.createSyntheticPSF(blur_px, yaw, psf);

        cv::Mat deblurred;
        deblurrer.wienerDeconvolution(blurred, psf, deblurred, 500.0);

        cv::imwrite(deblurredImage, deblurred);
        std::cout << "Deblurred image saved to: " << deblurredImage << "\n";

        return 0;
    } else {
        std::cerr << "Wrong arguments.\n";
        return 1;
    }
}

