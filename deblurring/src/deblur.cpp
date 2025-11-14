#include "metadata.h"
#include "helpers.h"

#include "../include/deblur.h"

#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

#ifdef DEBLUR_DEBUG
    #include "../include/debug.h"
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      // Errors
#define YELLOW  "\033[33m"      // Warnings
#define GREEN   "\033[32m"      // Success

namespace md = metadata;

Deblurrer::Deblurrer(DeblurConfig &config) { this->config_ = config; }

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

    const std::array<std::string, 6> texts = {"Text", "Complex Text", "Some More Complex Text", "It was blurred", "Can you read this?", "Top Secret"};
    
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

// Saves the deblurred image with all metadata of input image
void Deblurrer::saveImage(const cv::Mat &image, const fs::path& input, const std::string &prefix) {
    MEASURE_FUNCTION();
    fs::path output = constructPathWithPrefix(input, prefix);

    if (!config_.targetDir.empty()) {
        if (!fs::exists(config_.targetDir)) fs::create_directories(config_.targetDir);
        output = constructPathWithNewDir(output, config_.targetDir);
    }

    cv::imwrite(output, image);
    md::copyAll(input, output);

    fprintf(
        stdout,
        "[INFO] Final image is saved to: %s \r\n", output.c_str()
    );
}

// Saves the deblurred image with raw metadata
void Deblurrer::saveImage(const cv::Mat &image, const fs::path& input, const std::unordered_map<std::string, std::string>& md, const std::string &prefix) {
    MEASURE_FUNCTION();
    fs::path output = constructPathWithPrefix(input, prefix);

    if (!config_.targetDir.empty()) {
        if (!fs::exists(config_.targetDir)) fs::create_directories(config_.targetDir);
        output = constructPathWithNewDir(output, config_.targetDir);
    }

    cv::imwrite(output, image);
    md::copyAll(md, output);

    fprintf(
        stdout,
        "[INFO] Final image is saved to: %s \r\n", output.c_str()
    );
}

// Generates initial and blurred test images
void Deblurrer::generateTest(const fs::path& output) {
    if (!output.empty()) config_.testImagePath = output;

    cv::imwrite(config_.testImagePath, createTestImage());
    md::copyAll(md::getRandom(), config_.testImagePath);

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

    if (variance < blurThreshold) {
        std::fputs("[INFO] Image is likely blurred.\r\n", stdout);
        return true;
    } else {
        std::fputs("[INFO] Image is likely sharp.\r\n", stdout);
        return false;
    }
}

// Checks if image is blurred (by image path)
bool Deblurrer::isBlurred(const fs::path &imgpath, float blurThreshold = 100.0f) {
    cv::Mat image = cv::imread(imgpath, cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(
            stderr, 
            RED "[ERROR] Failed to load image from %s \r\n" RESET, imgpath.c_str()
        );
        return false;
    }
    return isBlurred(image, blurThreshold);
}

// Blur input image (works for both real and test-generated images)
void Deblurrer::blurImage(const fs::path& imgpath, bool grayscale) {
    MEASURE_FUNCTION();
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;

    cv::Mat normal = cv::imread(imgpath, imreadFlag);
    if (normal.empty()) {
        fprintf(
            stderr, 
            RED "[ERROR] Failed to load image: %s \r\n" RESET, imgpath.c_str()
        );
        return; 
    }

    if (config_.overwriteMetadata) md::copyAll(md::getRandom(), imgpath);

    auto md = md::extractAll(imgpath);

    int blurLength = 0;
    float blurAngleRad = 0.0f;
    findBlurLength(imgpath, blurLength, blurAngleRad, md);

    cv::Mat psf;
    estimatePSF(blurLength, blurAngleRad, psf);

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    saveImage(blurred, imgpath, md, "_blurred");
}

// Finds blur length by image metadata
void Deblurrer::findBlurLength(const fs::path& imgpath, int& blurLength, float& blurAngleRad, std::unordered_map<std::string, std::string>& md) { // px
    try {
        float exposure = 0.0f; md::tagAsFloat(md, "Exposure Time", exposure);
        float gsd = md::findGSD(md, config_.sensorWidth, config_.sensorHeight);
    
        float Vx, Vy, Vz, speed; findVBodies(md, Vx, Vy, Vz, speed);

        blurAngleRad = std::atan2(Vz, Vy);
    
        float blur = speed * 1000.0f * exposure;       // mm
        blurLength = static_cast<int>(blur / gsd);     // px
    
        fprintf(
            stdout, 
            "[INFO] Current blur length estimated: %.02f mm (%d px) \r\n", blur, blurLength
        );
    
        blurLength = (blurLength > 1) ? blurLength : 1; // px
    } catch (const std::exception& e) {
        md::listMetadata();
    }
}

void Deblurrer::findVBodies(const std::unordered_map<std::string, std::string>& md, float& Vx, float& Vy, float& Vz, float& speed) {
    float speedX, speedY, speedZ, yawRad, pitchRad, rollRad, exposure, gpsImgDirection;

    md::getPitchRollYawRad(md, pitchRad, rollRad, yawRad);
    md::getSpeedXYZ(md, speedX, speedY, speedZ); // in case where only GPS Speed is present, the result is stored in speedX

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
        gpsImgDirection = md::getGPSImgDirectionRad(md); // radians

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
        fputs(
            RED "[ERROR] PSF generation failed — normalization invalid. \r\n" RESET,
            stderr
        );
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

// works for odd/even; no cropping
void Deblurrer::fftshift(cv::Mat& input) {
    int cx = input.cols/2;
    int cy = input.rows/2;

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

    fftshift(paddedPSF); // As PSF is generated at the center of image, FFT Shift moves it to (0,0) for DFT
    
    cv::Mat psfDFT;
    cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);

    return psfDFT;
}

std::pair<cv::Mat, cv::Mat> Deblurrer::psfConjMag(const cv::Mat& psfDFT) {
    std::array<cv::Mat, 2> psfPlanes{};
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

inline const cv::Size Deblurrer::getDFTSize(cv::Size s) {
    const int H = cv::getOptimalDFTSize(s.height);
    const int W = cv::getOptimalDFTSize(s.width);
    return { W, H };
}

cv::Mat Deblurrer::padInput(const cv::Mat& input, const cv::Mat& psf) {
    cv::Mat inputF; input.convertTo(inputF, CV_32FC3, 1.0f/255.0f);

    const int padY = psf.rows;
    const int padX = psf.cols;
    cv::copyMakeBorder(inputF, inputF, padY, padY, padX, padX, cv::BORDER_REPLICATE);

    const cv::Size want = getDFTSize(inputF.size());
    const int addBottom = want.height - inputF.rows;
    const int addRight  = want.width  - inputF.cols;
    if (addBottom > 0 || addRight > 0) {
        cv::copyMakeBorder(inputF, inputF, 0, addBottom, 0, addRight, cv::BORDER_REPLICATE);
    }

    return inputF;
}

void Deblurrer::wienerDeconvolution(const cv::Mat& input, const cv::Mat& psf, cv::Mat& output, float blurLength, float snr) {
    MEASURE_FUNCTION();
    if (input.empty() || psf.empty()) {
        std::fputs(
            RED "[ERROR] Input or PSF is empty. \r\n" RESET, 
            stderr
        );
        return;
    }

    cv::Mat inputF = padInput(input, psf);
    cv::Mat psfDFT = psfdft(psf, inputF.size()); // expects normalized psf
    
    auto [psfConj, psfMag2] = psfConjMag(psfDFT);

    cv::Mat snrMap = createSNRMap(psfMag2.size(), psf);

    #ifdef DEBLUR_DEBUG
        visualizeMatrix(psfMag2, "psfMag2.png");
    #endif

    cv::Mat wienerDenom = buildWienerDenominator(psfMag2, snrMap, snr, inputF, psf);

    #ifdef DEBLUR_DEBUG
        visualizeMatrix(snrMap, "snrMap.png");
        visualizeMatrix(wienerDenom, "wienerDenom.png");
    #endif

    auto inputChannels = splitInputChannels(inputF);
    cv::Mat hann = createHannWindow2D(inputF.rows, inputF.cols);

    std::array<cv::Mat, 3> outputChannels = deconvolve(inputChannels, psfConj, wienerDenom, hann, input, psf);

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

std::array<cv::Mat, 3> Deblurrer::splitInputChannels(const cv::Mat& inputF) {
    std::array<cv::Mat, 3> inputChannels;
    if (inputF.channels() == 1)
        inputChannels[0] = inputF;
    else 
        cv::split(inputF, inputChannels);
    return inputChannels;
}

std::array<cv::Mat, 3> Deblurrer::deconvolve(const std::array<cv::Mat, 3>& inputChannels, const cv::Mat& psfConj, const cv::Mat& wienerDenom, const cv::Mat& hann, const cv::Mat& originalInput, const cv::Mat& psf) {
    std::array<cv::Mat, 3> outputChannels{};
    int padY = psf.rows;
    int padX = psf.cols;

    cv::parallel_for_(cv::Range(0, static_cast<int>(inputChannels.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Mat imgDFT;
            cv::dft(inputChannels[i].mul(hann), imgDFT, cv::DFT_COMPLEX_OUTPUT);

            cv::Mat filtered;
            cv::mulSpectrums(imgDFT, psfConj, filtered, 0);

            std::array<cv::Mat, 2> fPlanes;
            cv::split(filtered, fPlanes);

            #ifdef DEBLUR_DEBUG
                visualizeMagnitude(filtered, "filtered.png");
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

            cv::min(restored, 1.0f, restored);
            cv::max(restored, 0.0f, restored);

            outputChannels[i] = restored;
        }
    });

    return outputChannels;
}

void Deblurrer::denoiseImage(cv::Mat& image, float strength = 7.0f, float edgeStrength = 0.4f) {
    MEASURE_FUNCTION();
    if (image.empty()) {
        std::fputs(
            RED "[ERROR] Denoise input image is empty.\r\n" RESET, 
            stderr
        );
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
    }
}

void Deblurrer::deblurImage(const fs::path& imgpath, float snr = 1500.0) {
    MEASURE_FUNCTION();
    cv::Mat blurred = cv::imread(imgpath);
    if (blurred.empty()) {
        fprintf(
            stderr,
            RED "[ERROR] Failed to load image: %s \r\n" RESET, imgpath.c_str()
        );
        return;
    }

    if (!config_.forceDeblurring) {
        if (!isBlurred(blurred, config_.blurThreshold)) {
            fprintf(
                stdout,
                "[INFO] The image %s is normal. Skipping deblurring. \r\n", imgpath.c_str()
            );
            return;
        }
    }    

    auto md = md::extractAll(imgpath);

    int blurLength = 0;
    float blurAngleRad = 0.0f;
    findBlurLength(imgpath, blurLength, blurAngleRad, md);

    if (!blurLength) {
        fputs(
            RED "[ERROR] Failed to estimate blur length. \r\n" RESET,
            stderr
        );
        return;
    }

    cv::Mat psf; estimatePSF(blurLength, blurAngleRad, psf); // normalized automatically

    cv::Mat deblurred; wienerDeconvolution(blurred, psf, deblurred, blurLength, snr);

    if (config_.denoise) {
        if (!deblurred.empty()) {
            fputs(
                "[INFO] Applying post-deconvolution denoising. \r\n",
                stderr
            );
            denoiseImage(deblurred);
        } else {
            fputs(
                RED "[ERROR] Output image is empty. Skipping denoising and recovery step. \r\n" RESET,
                stderr
            );
        }
    }

    saveImage(deblurred, imgpath, md, "_deblurred"); // faster that copying from img to img
}