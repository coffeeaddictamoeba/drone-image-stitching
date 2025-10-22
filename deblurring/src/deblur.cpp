#include "metadata.h"
#include "helpers.h"

#include "../include/deblur.h"

#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <unordered_map>

#ifdef DEBLUR_DEBUG
    #include "../include/debug.h"
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success


namespace fs = std::filesystem;

Deblurrer::Deblurrer(DeblurConfig &config) { this->config_ = config; }


// -------------------- TEST IMAGE GENERATION AND BLUR -----------------------

// Creates random test image
cv::Mat Deblurrer::createTestImage(int width = 640, int height = 480) {
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> colorDist(0, 255);
    std::uniform_int_distribution<int> posXDist(0, width - 1);
    std::uniform_int_distribution<int> posYDist(0, height - 1);
    std::uniform_int_distribution<int> radiusDist(10, 60);
    std::uniform_int_distribution<int> thicknessDist(1, 5);
    std::uniform_int_distribution<int> lineLengthDist(30, 200);
    std::uniform_int_distribution<int> fontFaceDist(0, 7);
    std::uniform_real_distribution<double> fontScaleDist(0.5, 2.0);
    std::uniform_int_distribution<int> textThicknessDist(1, 3);

    // Lines
    for (int i = 0; i < 10; ++i) {
        cv::Point pt1(posXDist(gen), posYDist(gen));
        int length = lineLengthDist(gen);
        double angle = std::uniform_real_distribution<double>(0, 2 * CV_PI)(gen);
        cv::Point pt2(pt1.x + int(length * cos(angle)), pt1.y + int(length * sin(angle)));
        cv::line(canvas, pt1, pt2, cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen)), thicknessDist(gen), cv::LINE_AA);
    }

    // Circles
    for (int i = 0; i < 5; ++i) {
        cv::Point center(posXDist(gen), posYDist(gen));
        int radius = radiusDist(gen);
        int thickness = (i % 2 == 0) ? -1 : thicknessDist(gen);
        cv::circle(canvas, center, radius, cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen)), thickness, cv::LINE_AA);
    }

    // Rectangles
    for (int i = 0; i < 5; ++i) {
        cv::Point pt1(posXDist(gen), posYDist(gen));
        cv::Point pt2(pt1.x + lineLengthDist(gen) / 2, pt1.y + lineLengthDist(gen) / 3);
        int thickness = (i % 2 == 0) ? -1 : thicknessDist(gen);
        cv::rectangle(canvas, pt1, pt2, cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen)), thickness, cv::LINE_AA);
    }

    // Ellipses
    for (int i = 0; i < 5; ++i) {
        cv::Point center(posXDist(gen), posYDist(gen));
        cv::Size axes(radiusDist(gen), radiusDist(gen)/2);
        double angle = std::uniform_real_distribution<double>(0, 360)(gen);
        int thickness = thicknessDist(gen);
        cv::ellipse(canvas, center, axes, angle, 0, 360, cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen)), thickness, cv::LINE_AA);
    }

    std::vector<std::string> texts = {"Text", "Complex Text", "Some More Complex Text", "It was blurred", "Can you read this?", "Top Secret"};
    
    for (int i = 0; i < 7; ++i) {
        std::string text = texts[i % texts.size()];
        int fontFace = fontFaceDist(gen);
        double fontScale = fontScaleDist(gen);
        int thickness = textThicknessDist(gen);
        cv::Point org(posXDist(gen), posYDist(gen));
        cv::putText(canvas, text, org, fontFace, fontScale, cv::Scalar(colorDist(gen), colorDist(gen), colorDist(gen)), thickness, cv::LINE_AA);
    }

    return canvas;
}

// Creates random metadata to assign
std::unordered_map<std::string, std::string> Deblurrer::createTestMetadata() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> altitudeDist(50.0, 250.0);            // meters
    std::uniform_real_distribution<> focalLengthDist(2.0, 5.0);            // mm
    std::uniform_real_distribution<> speedDist(10.0, 25.0);                // km/h
    std::uniform_real_distribution<> speedXDist(10.0, 25.0);               // m/s
    std::uniform_real_distribution<> speedYDist(10.0, 25.0);               // m/s
    std::uniform_real_distribution<> speedZDist(10.0, 25.0);               // m/s
    std::uniform_real_distribution<> yawDist(0.0, 360.0);                  // degrees
    std::uniform_real_distribution<> pitchDist(0.0, 360.0);                // degrees
    std::uniform_real_distribution<> rollDist(0.0, 360.0);                 // degrees
    std::uniform_real_distribution<> exposureDist(1.0 / 10.0, 1.0 / 5.0);  // seconds

    float altitude = altitudeDist(gen);
    float focalLength = focalLengthDist(gen);
    float speed = speedDist(gen);
    float speedX = speedXDist(gen);
    float speedY = speedYDist(gen);
    float speedZ = speedZDist(gen);
    float yaw = yawDist(gen);
    float pitch = pitchDist(gen);
    float roll = rollDist(gen);
    float exposure = exposureDist(gen);

    std::unordered_map<std::string, std::string> metadata = {
        { "GPS Altitude", std::to_string(altitude) },
        { "GPS Speed", std::to_string(speed) },
        { "GPS Speed Ref", "km/h"},
        { "XMP-drone-dji:Flight X Speed", std::to_string(speedX) },
        { "XMP-drone-dji:Flight Y Speed", std::to_string(speedY) },
        { "XMP-drone-dji:Flight Z Speed", std::to_string(speedZ) },
        { "Focal Length", std::to_string(focalLength) },
        { "Flight Yaw Degree", std::to_string(yaw) },
        { "Flight Pitch Degree", std::to_string(pitch) },
        { "Flight Roll Degree", std::to_string(roll) },
        { "Exposure Time", std::to_string(exposure) }
    };

    #ifdef DEBUG
    std::cout << "[Info] Randomized metadata:\n";
    for (const auto& kv : metadata) {
        std::cout << "  " << kv.first << " = " << kv.second << "\n";
    }
    #endif

    return metadata;
}

void Deblurrer::saveImage(const cv::Mat &image, const std::string &inputImagePath, const std::string &prefix) {
    std::string newImagePath = constructPathWithPrefix(inputImagePath, prefix);

    if (!config_.targetDir.empty()) {
        if (!fs::exists(config_.targetDir)) fs::create_directories(config_.targetDir);
        newImagePath = constructPathWithNewDir(newImagePath, config_.targetDir);
    }

    cv::imwrite(newImagePath, image);
    copyMetadata(inputImagePath, newImagePath);

    std::cout << "[Info] Final image is saved to: " << newImagePath << "\n";
}

// Generates initial and blurred test images
void Deblurrer::generateTest(const std::string &testOutputPath) {
    if (!testOutputPath.empty()) config_.testImagePath = testOutputPath;

    cv::Mat synthetic = createTestImage();
    cv::imwrite(config_.testImagePath, synthetic);

    std::unordered_map<std::string, std::string> metadata = createTestMetadata();
    assignMetadata(config_.testImagePath, metadata);

    blurImage(config_.testImagePath, false);
}

// Checks if image is blurred (by image matrix)
bool Deblurrer::isBlurred(const cv::Mat &image, float blurThreshold = 100.0f, int maxImageSize = 1024) {
    cv::Mat resized;
    if (image.cols > maxImageSize || image.rows > maxImageSize) {
        float scale = maxImageSize / float(std::max(image.cols, image.rows));
        cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);
    } else {
        resized = image;
    }

    cv::Mat grayImage;
    if (resized.channels() == 3) {
        cv::cvtColor(resized, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        resized.copyTo(grayImage);
    }

    cv::Mat laplacianImage;
    cv::Laplacian(grayImage, laplacianImage, CV_32F);

    cv::Scalar mean, stdDev;
    cv::meanStdDev(laplacianImage, mean, stdDev);

    double variance = stdDev.val[0] * stdDev.val[0];

    #ifdef DEBUG
        std::cout << "[Info] Blur detection: Variance of Laplacian estimated: " << variance << "\n";
    #endif

    if (variance < blurThreshold) {
        std::cout << "[Info] Image is likely blurred." << std::endl;
        return true;
    } else {
        std::cout << "[Info] Image is likely sharp." << std::endl;
        return false;
    }
}

// Checks if image is blurred (by image path)
bool Deblurrer::isBlurred(const std::string &imagePath, float blurThreshold = 100.0f) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << RED << "[Error] Failed to load image from " << imagePath << RESET << std::endl;
        return false;
    }
    return isBlurred(image, blurThreshold);
}

// Blur input image (works for both real and test-generated images)
void Deblurrer::blurImage(const std::string &inputImagePath, bool grayscale) {
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;

    cv::Mat normal = cv::imread(inputImagePath, imreadFlag);
    if (normal.empty()) {
        std::cerr << RED << "[Error] Failed to load image: " << inputImagePath << RESET << std::endl;
        return; 
    }

    if (config_.overwriteMetadata) {
        auto metadata = createTestMetadata();
        assignMetadata(inputImagePath, metadata);
    }

    float blurAngleRad;
    float blurLength = findBlurLength(inputImagePath, blurAngleRad);

    cv::Mat psf;
    estimatePSF(blurLength, blurAngleRad, psf);

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    saveImage(blurred, inputImagePath, "_blurred");
}


// -------------------- IMAGE DEBLURRING -----------------------

// Finds blur length by image metadata
float Deblurrer::findBlurLength(const std::string &imagePath, float &blurAngleRad) { // px
    auto metadata = extractImageMetadata(imagePath);

    try {
        // Exposure Time in EXIF format most of the time looks like "1/10", so it is important to parse it properly
        float exposure = parseExifExposureTime(metadata["Exposure Time"]);
        float gsd = findGSD(metadata, config_.sensorWidth, config_.sensorHeight);
    
        float Vx, Vy, Vz, speed;
        findVBodies(metadata, Vx, Vy, Vz, speed);
    
        blurAngleRad = std::atan2(Vz, Vy);
    
        float blur = speed * 1000.0f * exposure; // mm
        int blurLength = static_cast<int>(blur / gsd); // px
    
        std::cout << "[Info] Current blur length estimated: " << blur << " mm " << "(" << blurLength << " px)" << std::endl; 
    
        return std::max(1, blurLength); // px
    } catch (const std::exception& e) {
        listMetadata();
        return 0.0f;
    }
}

void Deblurrer::findVBodies(const std::unordered_map<std::string, std::string> &metadata, float &Vx, float &Vy, float &Vz, float &speed) {
    float speedX, speedY, speedZ, yawRad, pitchRad, rollRad, exposure, gpsImgDirection;

    getPitchRollYawDeg(metadata, pitchRad, rollRad, yawRad);
    getSpeedXYZ(metadata, speedX, speedY, speedZ); // in case where only GPS Speed is present, the result is stored in speedX

    float cy = std::cos(yawRad);   float sy = std::sin(yawRad);
    float cp = std::cos(pitchRad); float sp = std::sin(pitchRad);
    float cr = std::cos(rollRad);  float sr = std::sin(rollRad);

    // This is the rotation matrix from World (NED) to Body frame (ZYX Euler sequence):
    // R = Rx(roll) * Ry(pitch) * Rz(yaw)
    // Vbody = R * VNED, where VNED = [speedXEast, speedYNorth, speedZDown]^T
            
    // Vx body - (forward/optical axis component) -> low influence on blur
    // Vy body - (right component in body frame, perpendicular to optical axis)
    // Vz body - (down component in body frame, perpendicular to optical axis)

    if (std::abs(speedX) > 1e-6f || std::abs(speedY) > 1e-6f || std::abs(speedZ) > 1e-6f) {
        // Vx = speedX * (cp * cy) + speedY * (cp * sy) + speedZ * (-sp); // not used in calculations
        Vy = speedX * (sr * sp * cy - cr * sy) + speedY * (sr * sp * sy + cr * cy) + speedZ * (sr * cp);
        Vz = speedX * (cr * sp * cy + sr * sy) + speedY * (cr * sp * sy - sr * cy) + speedZ * (cr * cp);
        speed = std::sqrt(Vy * Vy + Vz * Vz);
    } else {
        gpsImgDirection = getGPSImgDirectionDeg(metadata); // radians

        float vx = std::cos(gpsImgDirection);
        float vy = std::sin(gpsImgDirection);
        
        Vy = vx * (sr * sp * cy - cr * sy) + vy * (sr * sp * sy + cr * cy);
        Vz = vx * (cr * sp * cy + sr * sy) + vy * (cr * sp * sy - sr * cy);
        speed = speedX;
    }
}

// Estimate point spread function (PSF)
void Deblurrer::estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat& psf) {
    blurLengthPx = std::max(1, blurLengthPx);
    int ksize = std::max(blurLengthPx * 2 + 1, 15) | 1;  // Ensure odd

    psf = cv::Mat::zeros(ksize, ksize, CV_32F);
    const cv::Point2f center(ksize * 0.5f, ksize * 0.5f);

    float dx = std::cos(blurAngleRad);
    float dy = std::sin(blurAngleRad);
    float halfLen = 0.5f * blurLengthPx;

    cv::Point2f pt1(center.x - dx * halfLen, center.y - dy * halfLen);
    cv::Point2f pt2(center.x + dx * halfLen, center.y + dy * halfLen);

    cv::line(psf, pt1, pt2, cv::Scalar(1.0f), 1, cv::LINE_AA);

    if (blurLengthPx > 2) {
        int blurSize = ((blurLengthPx > 20) ? 5 : 3) | 1;
        float sigma = (blurLengthPx > 20) ? 1.0f : 0.3f;
        cv::GaussianBlur(psf, psf, cv::Size(blurSize, blurSize), sigma, sigma, cv::BORDER_REPLICATE);
    }

    #ifdef DEBLUR_DEBUG 
        visualizeMatrix(psf, "psf.png");
    #endif

    // Normalize PSF
    float normSum = static_cast<float>(cv::sum(psf)[0]);
    if (normSum > 1e-6f) {
        psf /= normSum;
    } else {
        std::cerr << RED << "[Error] PSF generation failed — normalization invalid." << RESET << "\n";
        psf.setTo(0);
    }
}

cv::Mat createHannWindow2D(int rows, int cols) {
    if (rows < 2 || cols < 2) {
        return cv::Mat::ones(std::max(1, rows), std::max(1, cols), CV_32F);
    }

    cv::Mat hannX(1, cols, CV_32F);
    cv::Mat hannY(rows, 1, CV_32F);

    for (int i = 0; i < cols; ++i)
        hannX.at<float>(0, i) = 0.5f * (1.0f - std::cos(2.0f * CV_PI * i / (cols - 1)));

    for (int i = 0; i < rows; ++i)
        hannY.at<float>(i, 0) = 0.5f * (1.0f - std::cos(2.0f * CV_PI * i / (rows - 1)));

    return hannY * hannX;  // outer product to form 2D window
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

cv::Mat Deblurrer::psfdft(const cv::Mat& normPSF, cv::Size targetSize) {
    cv::Mat paddedPSF(targetSize, CV_32F);
    paddedPSF.setTo(0);

    int cx = (paddedPSF.cols - normPSF.cols) / 2;
    int cy = (paddedPSF.rows - normPSF.rows) / 2;

    normPSF.copyTo(paddedPSF(cv::Rect(cx, cy, normPSF.cols, normPSF.rows)));

    fftShift(paddedPSF); // As PSF is generated at the center of image, FFT Shift moves it to (0,0) for DFT
    
    cv::Mat psfDFT;
    cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);
    return psfDFT;
}

std::pair<cv::Mat, cv::Mat> Deblurrer::psfConjMag(const cv::Mat& psfDFT) {
    std::vector<cv::Mat> psfPlanes(2);
    cv::split(psfDFT, psfPlanes);

    cv::Mat temp0, temp1, psfMag2;
    temp0 = psfPlanes[0].mul(psfPlanes[0]);
    temp1 = psfPlanes[1].mul(psfPlanes[1]);
    cv::add(temp0, temp1, psfMag2);

    double psfMaxVal;
    cv::minMaxLoc(psfMag2, nullptr, &psfMaxVal);
    cv::divide(psfMag2, static_cast<float>(psfMaxVal + 1e-6f), psfMag2);

    psfPlanes[1] = psfPlanes[1].mul(-1);

    cv::Mat psfConj;
    cv::merge(psfPlanes, psfConj);

    return {psfConj, psfMag2};
}

cv::Mat Deblurrer::padInput(const cv::Mat& input, const cv::Mat& psf) {
    cv::Mat inputF;
    input.convertTo(inputF, CV_32F, 1.0 / 255.0);

    int padY = psf.rows;
    int padX = psf.cols;
    cv::copyMakeBorder(inputF, inputF, padY, padY, padX, padX, cv::BORDER_REFLECT_101);

    return inputF;
}

void Deblurrer::wienerDeconvolution(const cv::Mat& input, const cv::Mat& psf, cv::Mat& output, float blurLength, float snr) {
    if (input.empty() || psf.empty()) {
        std::cerr << RED << "[Error] Input or PSF is empty." << RESET << std::endl;
        return;
    }

    // As GPU acceleration in this case is negligible, it is removed for now
    //cv::ocl::setUseOpenCL(true);

    cv::Mat inputF = padInput(input, psf);
    cv::Mat psfDFT = psfdft(psf, inputF.size()); // expects normalized psf
    
    auto [psfConj, psfMag2] = psfConjMag(psfDFT); // <cv::Mat, cv::Mat>

    cv::Mat snrMap = createSNRMap(inputF.size(), psf);
    //cv::Mat inputF_cpu; inputF.copyTo(inputF_cpu);          // cv::Mat -> cv::Mat
    //cv::Mat psfMag2_cpu; psfMag2.copyTo(psfMag2_cpu);       // cv::Mat -> cv::Mat
    #ifdef DEBLUR_DEBUG
        visualizeMatrix(psfMag2, "psfMag2.png");
        countMatrixZeros(psfMag2, "psfMag");
    #endif

    cv::Mat wienerDenom = buildWienerDenominator(psfMag2, snrMap, snr, inputF, psf);

    #ifdef DEBLUR_DEBUG
        visualizeMatrix(snrMap, "snrMap.png");
        visualizeMatrix(wienerDenom, "wienerDenom.png");
        countMatrixZeros(wienerDenom, "wienerDenom");
    #endif

    auto inputChannels = splitInputChannels(inputF);
    cv::Mat hann = createHannWindow2D(inputF.rows, inputF.cols);
    cv::Mat mask;
    if (blurLength >= 75.0f) {
        mask = createFeatherMask(input.size());
    } else {
        mask = cv::Mat(input.size(), CV_32F, 1.0f);
    }

    //cv::Mat psfConj_cpu; psfConj.copyTo(psfConj_cpu);
    auto outputChannels = deconvolve(inputChannels, psfConj, wienerDenom, hann, mask, input, psf);

    if (outputChannels.size() == 1)
        output = outputChannels[0];
    else
        cv::merge(outputChannels, output);

    output.convertTo(output, CV_8U, 255.0);
}

cv::Mat Deblurrer::createSNRMap(cv::Size size, const cv::Mat& psf) {
    cv::Mat snrMap(size, CV_32F);
    cv::Point center(size.width / 2, size.height / 2);
    float maxDist = std::sqrt(center.x * center.x + center.y * center.y);
    float blurLength = std::sqrt(psf.cols * psf.cols + psf.rows * psf.rows);
    float minFactor = std::clamp(0.7f - 0.015f * blurLength, 0.3f, 0.7f);

    cv::parallel_for_(cv::Range(0, snrMap.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float* rowPtr = snrMap.ptr<float>(y);
            for (int x = 0; x < snrMap.cols; ++x) {
                float dx = x - center.x;
                float dy = y - center.y;
                float dist = std::sqrt(dx * dx + dy * dy) / maxDist;
                float weight = std::cos(dist * CV_PI / 2.0f);
                rowPtr[x] = minFactor + (1.0f - minFactor) * weight;
            }
        }
    });

    return snrMap;
}

cv::Mat Deblurrer::buildWienerDenominator(const cv::Mat& psfMag2, const cv::Mat& snrMap, float snr, const cv::Mat& inputF, const cv::Mat& psf) {
    cv::Mat snrWeight = 1.0f / (snrMap * snr + 1e-6f);
    cv::Mat wienerDenom = psfMag2 + snrWeight;

    cv::threshold(wienerDenom, wienerDenom, 1e-5f, 1.0f, cv::THRESH_TOZERO);

    cv::Point center(inputF.cols / 2, inputF.rows / 2);
    cv::Point psfCenter(psf.cols / 2, psf.rows / 2);
    cv::Point blurVec(psf.cols - 1 - psfCenter.x, psf.rows - 1 - psfCenter.y);

    cv::Mat freqSuppression(inputF.size(), CV_32F, 1.0f);
    float maxFreq = std::sqrt(center.x * center.x + center.y * center.y);

    cv::parallel_for_(cv::Range(0, inputF.rows), [&](const cv::Range& range) {
        for (int y = range.start; y < range.end; ++y) {
            float* rowPtr = freqSuppression.ptr<float>(y);
            for (int x = 0; x < inputF.cols; ++x) {
                float dx = x - center.x;
                float dy = y - center.y;
                float dot = (dx * blurVec.x + dy * blurVec.y) / maxFreq;
                float angleFactor = 1.0f - 0.25f * std::abs(dot / maxFreq);
                rowPtr[x] = std::clamp(angleFactor, 0.7f, 1.0f);
            }
        }
    });

    wienerDenom /= freqSuppression;

    float blurLength = std::sqrt(psf.cols * psf.cols + psf.rows * psf.rows);
    float baseEps = std::clamp(0.0001f * blurLength, 1e-4f, 1e-3f);
    cv::max(wienerDenom, baseEps, wienerDenom);

    return wienerDenom;
}

std::vector<cv::Mat> Deblurrer::splitInputChannels(const cv::Mat& inputF) {
    std::vector<cv::Mat> inputChannels;
    if (inputF.channels() == 1)
        inputChannels.push_back(inputF);
    else
        cv::split(inputF, inputChannels);
    return inputChannels;
}

// causes blurring of image edges to hide artifacts in case of strong blur. higly questionable, but may be useful for blurLength <= 100 px + stitching
cv::Mat Deblurrer::createFeatherMask(cv::Size size) {
    cv::Mat mask(size, CV_32F, 1.0f);
    int feather = std::min(60, std::min(size.width, size.height) / 10);
    cv::rectangle(mask, cv::Rect(0, 0, size.width, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, size.height - feather, size.width, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, 0, feather, size.height), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(size.width - feather, 0, feather, size.height), 0.0f, -1);
    cv::GaussianBlur(mask, mask, cv::Size(2 * feather + 1, 2 * feather + 1), feather);
    return mask;
}

std::vector<cv::Mat> Deblurrer::deconvolve(const std::vector<cv::Mat>& inputChannels, const cv::Mat& psfConj, const cv::Mat& wienerDenom, const cv::Mat& hann, const cv::Mat& mask, const cv::Mat& originalInput, const cv::Mat& psf) {
    std::vector<cv::Mat> outputChannels(inputChannels.size());
    int padY = psf.rows;
    int padX = psf.cols;

    cv::parallel_for_(cv::Range(0, static_cast<int>(inputChannels.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Mat chWin = inputChannels[i].mul(hann);

            cv::Mat imgDFT;
            cv::dft(chWin, imgDFT, cv::DFT_COMPLEX_OUTPUT);

            cv::Mat filtered;
            cv::mulSpectrums(imgDFT, psfConj, filtered, 0);

            std::vector<cv::Mat> fPlanes(2);
            cv::split(filtered, fPlanes);

            #ifdef DEBLUR_DEBUG
                visualizeMagnitude(filtered, "filtered.png");
                countMatrixZeros(filtered, "filtered");
            #endif

            cv::Mat safeMask = (wienerDenom > 1e-3f);
            fPlanes[0].setTo(0, ~safeMask);
            fPlanes[1].setTo(0, ~safeMask);

            fPlanes[0] /= wienerDenom;
            fPlanes[1] /= wienerDenom;

            cv::merge(fPlanes, filtered);

            cv::Mat restored;
            cv::idft(filtered, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            restored /= hann + 1e-4f;

            restored = restored(cv::Rect(padX, padY, originalInput.cols, originalInput.rows));
            cv::Mat origCrop = inputChannels[i](cv::Rect(padX, padY, originalInput.cols, originalInput.rows));
            restored = restored.mul(mask) + origCrop.mul(1.0f - mask);

            cv::min(restored, 1.0f, restored);
            cv::max(restored, 0.0f, restored);

            #ifdef DEBLUR_DEBUG
                visualizeMatrix(restored, "restored.png");
                countMatrixZeros(restored, "restored");
            #endif

            outputChannels[i] = restored;
        }
    });

    return outputChannels;
}

// Possibly a solution for removing ghosting artifacts
// cv::Mat findLowContrastRegions(const cv::Mat &deblurred, const cv::Mat &psf, int blurLength) {
//     cv::Mat gray; 
//     if (deblurred.channels() == 3) 
//         cv::cvtColor(deblurred, gray, cv::COLOR_BGR2GRAY); 
//     else gray = deblurred.clone(); 
//     gray.convertTo(gray, CV_32F, 1.0 / 255.0); 

//     cv::Mat gray8u; gray.convertTo(gray8u, CV_8U, 255.0); 
//     cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8)); 
//     cv::Mat enhanced; 
//     clahe->apply(gray8u, enhanced); 
//     enhanced.convertTo(enhanced, CV_32F, 1.0 / 255.0); 

//     cv::Moments m = cv::moments(psf, true); 
//     double angle = 0.0; 
    
//     if (m.mu20 + m.mu02 != 0) angle = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02); // radians 

//     cv::Point2f shift(std::cos(angle), std::sin(angle)); 
//     cv::Mat shifted = cv::Mat::zeros(enhanced.size(), CV_32F); 
//     cv::Mat M = (cv::Mat_<float>(2, 3) << 1, 0, blurLength * shift.x, 0, 1, blurLength * shift.y); 
//     cv::warpAffine(enhanced, shifted, M, enhanced.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT); 

//     cv::Mat diff; 
//     cv::absdiff(enhanced, shifted, diff);
//     cv::Mat ghostLikelihood; ghostLikelihood = 1.0 - diff; 
//     cv::normalize(ghostLikelihood, ghostLikelihood, 0, 1, cv::NORM_MINMAX); 

//     cv::GaussianBlur(ghostLikelihood, ghostLikelihood, cv::Size(7, 7), 2.0); 
//     cv::Mat mask8u; cv::normalize(ghostLikelihood, ghostLikelihood, 0, 255, cv::NORM_MINMAX); 
//     ghostLikelihood.convertTo(mask8u, CV_8U); 
//     return mask8u;
// }

void Deblurrer::denoiseImage(cv::Mat& image, float strength = 10.0f, float edgeStrength = 0.4f) {
    if (image.empty()) {
        std::cerr << RED << "[Error] Denoise input image is empty." << RESET << std::endl;
        return;
    }

    cv::Mat tempImage;
    if (image.depth() != CV_8U) {
        image.convertTo(tempImage, CV_8U, 255.0);
    } else {
        tempImage = image;
    }

    cv::edgePreservingFilter(tempImage, tempImage, cv::RECURS_FILTER, strength, edgeStrength);

    if (image.depth() != CV_8U) {
        tempImage.convertTo(image, image.type(), 1.0 / 255.0);
    } else {
        image = tempImage;
    }_blurred
}

void Deblurrer::deblurImage(const std::string &inputImagePath, float snr = 1500.0) {
    cv::Mat blurred = cv::imread(inputImagePath);
    if (blurred.empty()) {
        std::cerr << RED << "[Error] Failed to load image: " << inputImagePath << RESET << std::endl;
        return;
    }

    if (!config_.forceDeblurring) {
        if (!isBlurred(blurred, config_.blurThreshold)) {
            std::cout << "[Info] The image " + inputImagePath + " is normal. Skipping deblurring.\n";
            return;
        }
    }    

    float blurAngleRad;
    float blurLength = findBlurLength(inputImagePath, blurAngleRad);

    if (!blurLength) {
        std::cerr << RED << "[Error] Failed to estimate blur length." << RESET << std::endl;
        return;
    }

    cv::Mat psf;
    estimatePSF(blurLength, blurAngleRad, psf); // normalized automatically

    cv::Mat deblurred;
    wienerDeconvolution(blurred, psf, deblurred, blurLength, snr);

    if (config_.denoise) {
        if (!deblurred.empty()) {
            std::cout << "[Info] Applying post-deconvolution denoising.\n";
            denoiseImage(deblurred);
        } else {
            std::cerr << YELLOW << "[Warn] Output image is empty. Skipping denoising and recovery step." << RESET << std::endl;
        }
    }

    saveImage(deblurred, inputImagePath, "_deblurred");

    // experinmental
    // cv::Mat lowContrast = findLowContrastRegions(deblurred, psf, blurLength);
    // cv::Mat final = removeGhosting(deblurred, lowContrast, blurLength, blurAngleRad);

    // final.convertTo(final, CV_8U);
    // saveImage(final, inputImagePath, "_ghostmask");
}