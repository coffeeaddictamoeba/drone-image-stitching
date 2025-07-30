#include "../include/deblur.h"
#include "../include/helpers.h"
#include "../include/metadata.h"
#include <cstring>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <unordered_map>

Deblurrer::Deblurrer(DeblurConfig &config) {
    this->config_ = config;
}


// -------------------- TEST IMAGE GENERATION AND BLUR -----------------------

// creates test image
cv::Mat Deblurrer::createTestImage(int width = 640, int height = 480) {
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    cv::line(canvas, cv::Point(50, 100), cv::Point(600, 100), cv::Scalar(0, 0, 0), 2);
    cv::line(canvas, cv::Point(50, 200), cv::Point(600, 200), cv::Scalar(0, 0, 0), 3);
    cv::line(canvas, cv::Point(100, 400), cv::Point(500, 100), cv::Scalar(0, 0, 255), 2);
    cv::line(canvas, cv::Point(300, 450), cv::Point(600, 250), cv::Scalar(255, 0, 0), 2);

    cv::circle(canvas, cv::Point(320, 240), 40, cv::Scalar(0, 255, 0), -1); // filled

    cv::putText(canvas, "Test Image", cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);

    return canvas;
}

// creates random metadata to assign
std::unordered_map<std::string, std::string> Deblurrer::createTestMetadata() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> altitudeDist(50.0, 150.0);       // meters
    std::uniform_real_distribution<> focalLengthDist(2.0, 5.0);       // mm
    std::uniform_real_distribution<> speedDist(15.0, 25.0);            // km/h
    std::uniform_real_distribution<> yawDist(0.0, 360.0);             // degrees
    std::uniform_real_distribution<> pitchDist(0.0, 360.0);             // degrees
    std::uniform_real_distribution<> rollDist(0.0, 360.0);             // degrees
    std::uniform_real_distribution<> exposureDist(1.0 / 10.0, 1.0 / 5.0); // seconds

    float altitude = altitudeDist(gen);
    float focalLength = focalLengthDist(gen);
    float speed = speedDist(gen);
    float yaw = yawDist(gen);
    float pitch = pitchDist(gen);
    float roll = rollDist(gen);
    float exposure = exposureDist(gen);

    std::unordered_map<std::string, std::string> metadata = {
        { "GPS Altitude", std::to_string(altitude) },
        { "GPS Speed", std::to_string(speed) },
        { "GPS Speed Ref", "km/h"},
        { "Focal Length", std::to_string(focalLength) },
        { "Flight Yaw Degree", std::to_string(yaw) },
        { "Flight Pitch Degree", std::to_string(pitch) },
        { "Flight Roll Degree", std::to_string(roll) },
        { "Exposure Time", std::to_string(exposure) }
    };

    std::cout << "Randomized metadata:\n";
    for (const auto& kv : metadata) {
        std::cout << "  " << kv.first << " = " << kv.second << "\n";
    }

    return metadata;
}

// generates initial and blurred test images
void Deblurrer::generateTest(const std::string &testOutputPath) {
    if (!testOutputPath.empty()) {
        config_.testImagePath = testOutputPath;
    }

    std::string prefix = "_blurred";
    std::string blurredImage = constructPathWithPrefix(config_.testImagePath, prefix);

    cv::Mat synthetic = createTestImage();
    cv::imwrite(config_.testImagePath, synthetic);

    std::unordered_map<std::string, std::string> metadata = createTestMetadata();
    assignMetadata(config_.testImagePath, metadata);

    blurImage(config_.testImagePath, blurredImage, false);
}

// checks if image is blurred (by image matrix)
bool Deblurrer::isBlurred(const cv::Mat &image, float blurThreshold = 100.0f) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    cv::Mat laplacianImage;
    cv::Laplacian(grayImage, laplacianImage, CV_64F);

    cv::Scalar mean, stdDev;
    cv::meanStdDev(laplacianImage, mean, stdDev);

    double variance = stdDev.val[0] * stdDev.val[0];

    std::cout << "[Info] Blur detection: Variance of Laplacian estimated: " << variance << "\n";

    if (variance < blurThreshold) {
        std::cout << "[Info] Image is likely blurred.\n";
        return true;
    } else {
        std::cout << "[Info] Image is likely sharp.\n";
        return false;
    }
}

// checks if image is blurred (by image path)
bool Deblurrer::isBlurred(const std::string &imagePath, float blurThreshold = 100.0f) {
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "[ERROR] isBlurred: Failed to load image from " << imagePath << "\n";
        return false;
    }

    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    cv::Mat laplacianImage;
    cv::Laplacian(grayImage, laplacianImage, CV_64F);

    cv::Scalar mean, stdDev;
    cv::meanStdDev(laplacianImage, mean, stdDev);

    double variance = stdDev.val[0] * stdDev.val[0];

    std::cout << "[Info] Blur detection: Variance of Laplacian estimated: " << variance << "\n";

    if (variance < blurThreshold) {
        std::cout << "[Info] Image is likely blurred.\n";
        return true;
    } else {
        std::cout << "[Info] Image is likely sharp.\n";
        return false;
    }
}

// blur input image (works for both real and test-generated images)
void Deblurrer::blurImage(const std::string &inputImagePath, const std::string &outputImagePath, bool grayscale) {
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat normal = cv::imread(inputImagePath, imreadFlag);
    if (normal.empty()) {
        std::cerr << "Failed to load image: " << inputImagePath << "\n";
        return; 
    }

    float blurLength = findBlurLength(inputImagePath);
    float yaw = std::stof(extractExifTagValue(inputImagePath, "FlightYawDegree"));

    cv::Mat psf;
    estimatePSF(blurLength, yaw, psf);

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    imwrite(outputImagePath, blurred);
    copyMetadata(inputImagePath, outputImagePath);

    std::cout << "Blurred image saved to: " << outputImagePath << "\n";
}


// -------------------- IMAGE DEBLURRING -----------------------

// finds blur length by image metadata
float Deblurrer::findBlurLength(const std::string &imagePath) { // px
    auto metadata = extractImageMetadata(imagePath);

    // suitable for tests or synthetic blurring, not recommended to use otherwise
    if (config_.overwriteMetadata) {
        metadata = createTestMetadata();
        assignMetadata(imagePath, metadata);
    }

    float yaw, speed, exposure;
    try {
        yaw = metadata.count("Flight Yaw Degree") ? std::stof(metadata["Flight Yaw Degree"]) : 0.0f;
        if (yaw == 0.0f) {
            std::cout << "[Warn] Yaw of " << yaw << " was detected. This can be an indicator of empty yaw metadata.\n";
        }
        speed = parseExifGPSSpeed(metadata["GPS Speed"], metadata["GPS Speed Ref"]);
        exposure = parseExifExposureTime(metadata["Exposure Time"]);
    } catch (...) {
        std::cerr << "[Error] Image lacks essential metadata. Please check these exiftool tags:\n" 
                << "    - FlightYawDegree\n"
                << "    - GPSSpeed\n"
                << "    - GPSSpeedRef\n"
                << "    - ExposureTime\n"
                << "    - GPSAltitude\n";
        return 0.0f;
    }

    float alt = std::stof(metadata["GPS Altitude"]); // m
    float flen = std::stof(metadata["Focal Length"]); // mm
    int imageWidth = std::stoi(metadata["Image Width"]); // px
    int imageHeight = std::stoi(metadata["Image Height"]); // px

    float gsd = calculateGSD(alt, flen, imageWidth, imageHeight, config_.sensorWidth, config_.sensorHeight); // mm/px
    float blur = speed * 1000.0f * exposure; // mm
    int blurLength = static_cast<int>(blur / gsd); // px

    std::cout << "Original metadata: speed = " << speed << " m/s, yaw = " << yaw << " deg, exposure = " << exposure << " s (" << blurLength << " px blur)" << std::endl; 

    return std::max(1, blurLength); // px
}

// estimate point spread function (PSF)
void Deblurrer::estimatePSF(int blurLengthPx, float yawDeg, cv::Mat& psf) {
    blurLengthPx = std::max(1, blurLengthPx);
    int ksize = std::max(blurLengthPx * 2 + 1, 15) | 1;  // ensure odd

    psf = cv::Mat::zeros(ksize, ksize, CV_32F);
    const cv::Point2f center(ksize * 0.5f, ksize * 0.5f);

    float angleRad = yawDeg * (CV_PI / 180.0f);
    float dx = std::cos(angleRad);
    float dy = std::sin(angleRad);
    float halfLen = 0.5f * blurLengthPx;

    cv::Point2f pt1(center.x - dx * halfLen, center.y - dy * halfLen);
    cv::Point2f pt2(center.x + dx * halfLen, center.y + dy * halfLen);

    cv::line(psf, pt1, pt2, cv::Scalar(1.0f), 1, cv::LINE_AA);

    if (blurLengthPx > 2) {
        int blurSize = ((blurLengthPx > 20) ? 5 : 3) | 1;
        float sigma = (blurLengthPx > 20) ? 1.0f : 0.3f;
        cv::GaussianBlur(psf, psf, cv::Size(blurSize, blurSize), sigma, sigma, cv::BORDER_REPLICATE);
    }

    double normSum = cv::sum(psf)[0];
    if (normSum <= 1e-6) {
        std::cerr << "[ERROR] PSF generation failed — normalization invalid.\n";
        psf.setTo(0);
        return;
    }

    psf /= static_cast<float>(normSum);

    #ifdef DEBUG
        cv::Mat debug;
        cv::normalize(psf, debug, 0, 255, cv::NORM_MINMAX);
        debug.convertTo(debug, CV_8U);
        cv::imwrite("psf.png", debug);
    #endif

    // Optional: remove for release
    std::cout << "PSF size: " << psf.cols << "x" << psf.rows << std::endl;
}

// calculate ground sample distance (GSD)
float Deblurrer::calculateGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth = 3.68f, float sensorHeight = 2.76f) {
    altitude *= 1000.0f; // m -> mm

    float gsdWidth = (altitude * sensorWidth) / (focalLength * imageWidth); // mm/px
    float gsdHeight = (altitude * sensorHeight) / (focalLength * imageHeight); // mm/px

    float gsd = std::max(gsdWidth, gsdHeight); // mm/px

    std::cout << "GSD: Calculated GSD = " << gsd << " mm/px\n";
    std::cout << "  Focal Length: " << focalLength << " mm\n";
    std::cout << "  Sensor Size: " << sensorWidth << "x" << sensorHeight << " mm\n";
    std::cout << "  Image Resolution: " << imageWidth << "x" << imageHeight << " px\n";
    std::cout << "  Altitude: " << altitude << " mm\n";

    return gsd;
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

void Deblurrer::wienerDeconvolution(const cv::Mat& input, const cv::Mat& psf, cv::Mat& output, float snr) {
    if (input.empty() || psf.empty()) {
        std::cerr << "[ERROR] Input or PSF is empty.\n";
        return;
    }

    // Enable OpenCL (GPU support if available)
    cv::ocl::setUseOpenCL(true);

    // Normalize input
    cv::Mat inputF;
    input.convertTo(inputF, CV_32F, 1.0 / 255.0);

    // Pad image
    int padY = psf.rows / 2;
    int padX = psf.cols / 2;
    cv::copyMakeBorder(inputF, inputF, padY, padY, padX, padX, cv::BORDER_REFLECT_101);

    // Normalize PSF
    cv::Mat normPSF;
    psf.convertTo(normPSF, CV_32F);
    normPSF /= static_cast<float>(cv::sum(normPSF)[0]);

    // Center PSF in padded matrix
    cv::Mat paddedPSF = cv::Mat::zeros(inputF.size(), CV_32F);
    int x = (paddedPSF.cols - normPSF.cols) / 2;
    int y = (paddedPSF.rows - normPSF.rows) / 2;
    normPSF.copyTo(paddedPSF(cv::Rect(x, y, normPSF.cols, normPSF.rows)));
    fftShift(paddedPSF); // Custom function to shift PSF center to corners

    // FFT of PSF
    cv::Mat psfDFT;
    cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);

    // Precompute conjugate and |H|^2 + 1/SNR
    std::vector<cv::Mat> psfPlanes(2);
    cv::split(psfDFT, psfPlanes);
    cv::Mat psfMag2;
    cv::magnitude(psfPlanes[0], psfPlanes[1], psfMag2);
    psfMag2 = psfMag2.mul(psfMag2) + (1.0f / snr) + 1e-6f;

    // Compute conjugate
    psfPlanes[1] *= -1;
    cv::Mat psfConj;
    cv::merge(psfPlanes, psfConj);

    // Split input into channels
    std::vector<cv::Mat> inputChannels;
    if (inputF.channels() == 1)
        inputChannels.push_back(inputF);
    else
        cv::split(inputF, inputChannels);

    // Shared Hann window
    cv::Mat hann = createHannWindow2D(inputF.rows, inputF.cols); // Custom function

    // Shared feathering mask
    cv::Mat mask(input.size(), CV_32F, 1.0f);
    int feather = std::min(30, std::min(input.cols, input.rows) / 10);
    cv::rectangle(mask, cv::Rect(0, 0, input.cols, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, input.rows - feather, input.cols, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, 0, feather, input.rows), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(input.cols - feather, 0, feather, input.rows), 0.0f, -1);
    cv::GaussianBlur(mask, mask, cv::Size(2 * feather + 1, 2 * feather + 1), feather);

    // Output container
    std::vector<cv::Mat> outputChannels(inputChannels.size());

    // Parallel deconvolution
    cv::parallel_for_(cv::Range(0, static_cast<int>(inputChannels.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Mat chWin = inputChannels[i].mul(hann);

            cv::Mat imgDFT;
            cv::dft(chWin, imgDFT, cv::DFT_COMPLEX_OUTPUT);

            // Multiply in frequency domain: imgDFT * psfConj
            cv::Mat filtered;
            cv::mulSpectrums(imgDFT, psfConj, filtered, 0);

            // Divide by Wiener denominator (magnitude squared + 1/SNR)
            std::vector<cv::Mat> fPlanes(2);
            cv::split(filtered, fPlanes);
            fPlanes[0] /= psfMag2;
            fPlanes[1] /= psfMag2;
            cv::merge(fPlanes, filtered);

            // Inverse DFT
            cv::Mat restored;
            cv::idft(filtered, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            // Undo Hann
            restored /= hann + 1e-4f;

            // Crop and feather blend
            restored = restored(cv::Rect(padX, padY, input.cols, input.rows));
            cv::Mat origCrop = inputChannels[i](cv::Rect(padX, padY, input.cols, input.rows));
            restored = restored.mul(mask) + origCrop.mul(1.0f - mask);

            // Clamp to [0, 1]
            cv::min(restored, 1.0f, restored);
            cv::max(restored, 0.0f, restored);

            outputChannels[i] = restored;
        }
    });

    // Merge channels and convert
    cv::Mat merged;
    if (outputChannels.size() == 1)
        merged = outputChannels[0];
    else
        cv::merge(outputChannels, merged);

    merged.convertTo(output, CV_8U, 255.0);
}

void Deblurrer::denoiseImage(cv::Mat& image, float strength = 10.0f, float edgeStrength = 0.4f) {
    if (image.empty()) {
        std::cerr << "[Error] Denoise input image is empty.\n";
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

void Deblurrer::recoverBrightness(cv::Mat& image, float gamma) {
    if (image.empty()) {
        std::cerr << "[Error] Gamma correction input image is empty.\n";
        return;
    }

    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for( int i = 0; i < 256; ++i) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }

    cv::LUT(image, lookUpTable, image);
}

void Deblurrer::deblurImage(const std::string &inputImagePath, const std::string &outputImagePath, float snr = 500.0) {
    cv::Mat blurred = cv::imread(inputImagePath);
    if (blurred.empty()) {
        std::cerr << "Failed to load image: " << inputImagePath << "\n";
        return;
    }

    if (!config_.forceDeblurring) {
        if (!isBlurred(blurred, config_.blurThreshold)) {
            std::cout << "[Info] The image " + inputImagePath + " is normal. Skipping deblurring.\n";
            return;
        }
    }    

    float blurLength = findBlurLength(inputImagePath);
    float yaw = std::stof(extractExifTagValue(inputImagePath, "FlightYawDegree"));

    cv::Mat psf;
    estimatePSF(blurLength, yaw, psf);

    cv::Mat deblurred;
    wienerDeconvolution(blurred, psf, deblurred, snr);

    if (config_.denoise) {
        if (!deblurred.empty()) {
            std::cout << "[Info] Applying post-deconvolution denoising.\n";
            denoiseImage(deblurred);
        } else {
            std::cerr << "[Warn] Output image is empty. Skipping denoising and recovery step.\n";
        }
    }

    cv::imwrite(outputImagePath, deblurred);
    copyMetadata(inputImagePath, outputImagePath);
    std::cout << "Deblurred image saved to: " << outputImagePath << "\n";
}