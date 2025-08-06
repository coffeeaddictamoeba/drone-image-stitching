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

Deblurrer::Deblurrer(DeblurConfig &config) { this->config_ = config; }


// -------------------- DEBUG HELPERS ----------------------------------------

void visualizeMatrix(cv::Mat image, std::string outputImagePath) {
    cv::Mat debug;
    cv::normalize(image, debug, 0, 255, cv::NORM_MINMAX);
    debug.convertTo(debug, CV_8U);
    cv::imwrite(outputImagePath, debug);
}

void visualizeMagnitude(cv::Mat complexImage, std::string outputImagePath) {
    std::vector<cv::Mat> planes;
    cv::split(complexImage, planes);

    cv::Mat mag;
    cv::magnitude(planes[0], planes[1], mag);
    mag += 1e-5f;

    cv::log(mag, mag);
    cv::normalize(mag, mag, 0, 255, cv::NORM_MINMAX);
    mag.convertTo(mag, CV_8U);

    visualizeMatrix(mag, outputImagePath);
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
    std::cout << "Randomized metadata:\n";
    for (const auto& kv : metadata) {
        std::cout << "  " << kv.first << " = " << kv.second << "\n";
    }
    #endif

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

// checks if image is blurred (by image path)
bool Deblurrer::isBlurred(const std::string &imagePath, float blurThreshold = 100.0f) {
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr << "[ERROR] isBlurred: Failed to load image from " << imagePath << std::endl;
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

// blur input image (works for both real and test-generated images)
void Deblurrer::blurImage(const std::string &inputImagePath, const std::string &outputImagePath, bool grayscale) {
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;

    cv::Mat normal = cv::imread(inputImagePath, imreadFlag);
    if (normal.empty()) {
        std::cerr << "Failed to load image: " << inputImagePath << std::endl;
        return; 
    }

    float blurAngleRad;
    float blurLength = findBlurLength(inputImagePath, blurAngleRad);

    cv::Mat psf;
    estimatePSF(blurLength, blurAngleRad, psf);

    cv::Mat blurred;
    filter2D(normal, blurred, -1, psf, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    imwrite(outputImagePath, blurred);
    copyMetadata(inputImagePath, outputImagePath);

    std::cout << "Blurred image saved to: " << outputImagePath << "\n";
}


// -------------------- IMAGE DEBLURRING -----------------------

// finds blur length by image metadata
float Deblurrer::findBlurLength(const std::string &imagePath, float &blurAngleRad) { // px
    // suitable for tests or synthetic blurring, not recommended to use otherwise
    if (config_.overwriteMetadata) {
        auto metadata = createTestMetadata();
        assignMetadata(imagePath, metadata);
    }

    auto metadata = extractImageMetadata(imagePath);

    bool use3DSpeed = false;
    float yaw, pitch, roll, speed, speedX, speedY, speedZ, exposure;
    
    float yawRad = 0.0f;
    float pitchRad = 0.0f;
    float rollRad = 0.0f;

    try {
        yaw = metadata.count("Flight Yaw Degree") ? std::stof(metadata["Flight Yaw Degree"]) : 0.0f;       // degrees
        pitch = metadata.count("Flight Pitch Degree") ? std::stof(metadata["Flight Pitch Degree"]) : 0.0f; // degrees
        roll = metadata.count("Flight Roll Degree") ? std::stof(metadata["Flight Roll Degree"]) : 0.0f;    // degrees

        yawRad = yaw * (CV_PI / 180.0f);
        pitchRad = pitch * (CV_PI / 180.0f);
        rollRad = roll * (CV_PI / 180.0f);

        speedX = metadata.count("Flight X Speed") ? std::stof(metadata["Flight X Speed"]) : 0.0f; // m/s (East)
        speedY = metadata.count("Flight Y Speed") ? std::stof(metadata["Flight Y Speed"]) : 0.0f; // m/s (North)
        speedZ = metadata.count("Flight Z Speed") ? std::stof(metadata["Flight Z Speed"]) : 0.0f; // m/s (Down)

        if ((std::abs(speedX) > 1e-6f || std::abs(speedY) > 1e-6f || std::abs(speedZ) > 1e-6f) && 
            metadata.count("Flight X Speed") && metadata.count("Flight Y Speed") && metadata.count("Flight Z Speed")) {
            #ifdef DEBUG
            std::cout << "[Info] Using 3D Speed and Orientation parameters:\n"
                      << "    - Speed X (East): " << speedX << " m/s\n"
                      << "    - Speed Y (North): " << speedY << " m/s\n"
                      << "    - Speed Z (Down): " << speedZ << " m/s\n"
                      << "    - Yaw: " << yaw << " deg\n"
                      << "    - Pitch: " << pitch << " deg\n"
                      << "    - Roll: " << roll << " deg\n";
            #endif
            use3DSpeed = true;
        } else {
            std::cout << "[Warn] Cannot find sufficient 3D Speed parameters. Using GPS Speed.\n";
            if (!metadata.count("GPS Speed") || !metadata.count("GPS Speed Ref")) {
                 throw std::runtime_error("Missing GPSSpeed or GPSSpeedRef for 2D speed fallback.");
            }
            speed = parseExifGPSSpeed(metadata["GPS Speed"], metadata["GPS SpeedRef"]);
        }
        
        if (!metadata.count("Exposure Time")) {
            throw std::runtime_error("Missing Exposure Time metadata.");
        }
        exposure = parseExifExposureTime(metadata["Exposure Time"]);

    } catch (const std::exception& e) {
        std::cerr << "[Error] Image lacks essential metadata or parsing failed: " << e.what() << "\n"
                << "    Please check these exiftool tags:\n"
                << "    - Flight Yaw Degree\n"
                << "    - Flight Pitch Degree\n"
                << "    - Flight Roll Degree\n"
                << "    - GPS Speed\n"
                << "    - GPS Speed Ref\n"
                << "    - GPS Altitude\n"
                << "    - Exposure Time\n"
                << " If you are using 3D speed parameters, also check:\n"
                << "    - Flight X Speed\n" 
                << "    - Flight Y Speed\n" 
                << "    - Flight Z Speed\n"; 
        return 0.0f;
    }
    
    float alt = std::stof(metadata["GPS Altitude"]);       // m
    float flen = std::stof(metadata["Focal Length"]);      // mm
    int imageWidth = std::stoi(metadata["Image Width"]);   // px
    int imageHeight = std::stoi(metadata["Image Height"]); // px

    float gsd = calculateGSD(alt, flen, imageWidth, imageHeight, config_.sensorWidth, config_.sensorHeight); // mm/px
    float blur = 0.0f;

    if (use3DSpeed) {
        float cy = std::cos(yawRad);
        float sy = std::sin(yawRad);

        float cp = std::cos(pitchRad);
        float sp = std::sin(pitchRad);

        float cr = std::cos(rollRad);
        float sr = std::sin(rollRad);

        // This is the rotation matrix from World (NED) to Body frame (ZYX Euler sequence):
        // R = Rx(roll) * Ry(pitch) * Rz(yaw)
        // Vbody = R * VNED, where VNED = [speedXEast, speedYNorth, speedZDown]^T
        
        // Vx body (forward/optical axis component) -> low influence on blur
        float Vx = speedX * (cp * cy) +
                    speedY * (cp * sy) +
                    speedZ * (-sp);

        // Vy body (right component in body frame, perpendicular to optical axis)
        float Vy = speedX * (sr * sp * cy - cr * sy) +
                    speedY * (sr * sp * sy + cr * cy) +
                    speedZ * (sr * cp);

        // Vz body (down component in body frame, perpendicular to optical axis)
        float Vz = speedX * (cr * sp * cy + sr * sy) +
                    speedY * (cr * sp * sy - sr * cy) +
                    speedZ * (cr * cp);

        float speed3D = std::sqrt(Vy * Vy + Vz * Vz);
        
        blur = speed3D * 1000.0f * exposure; // mm
        blurAngleRad = std::atan2(Vz, Vy);
    } else {
        blur = speed * 1000.0f * exposure; // mm
    }

    int blurLength = static_cast<int>(blur / gsd); // px

    #ifdef DEBUG
        std::cout << "[Info] Current blur length estimated: " << blur << " mm " << "(" << blurLength << " px)" << std::endl; 
    #endif

    return std::max(1, blurLength); // px
}

// estimate point spread function (PSF)
void Deblurrer::estimatePSF(int blurLengthPx, float blurAngleRad, cv::Mat& psf) {
    blurLengthPx = std::max(1, blurLengthPx);
    int ksize = std::max(blurLengthPx * 2 + 1, 15) | 1;  // ensure odd

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

    double normSum = cv::sum(psf)[0];
    if (normSum <= 1e-6) {
        std::cerr << "[ERROR] PSF generation failed — normalization invalid.\n";
        psf.setTo(0);
        return;
    }

    psf /= static_cast<float>(normSum);

    #ifdef DEBUG
        visualizeMatrix(psf, "psf.png");
        std::cout << "PSF size: " << psf.cols << "x" << psf.rows << std::endl;
    #endif    
}

// calculate ground sample distance (GSD)
float Deblurrer::calculateGSD(float altitude, float focalLength, int imageWidth, int imageHeight, float sensorWidth = 3.68f, float sensorHeight = 2.76f) {
    altitude *= 1000.0f; // m -> mm

    float gsdWidth = (altitude * sensorWidth) / (focalLength * imageWidth);    // mm/px
    float gsdHeight = (altitude * sensorHeight) / (focalLength * imageHeight); // mm/px

    float gsd = std::max(gsdWidth, gsdHeight); // mm/px

    #ifdef DEBUG
        std::cout << "GSD: Calculated GSD = " << gsd << " mm/px\n";
        std::cout << "  Focal Length: " << focalLength << " mm\n";
        std::cout << "  Sensor Size: " << sensorWidth << "x" << sensorHeight << " mm\n";
        std::cout << "  Image Resolution: " << imageWidth << "x" << imageHeight << " px\n";
        std::cout << "  Altitude: " << altitude << " mm\n";
    #endif

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

    cv::ocl::setUseOpenCL(true); // turn on GPU if available

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
    int cx = (paddedPSF.cols - normPSF.cols) / 2;
    int cy = (paddedPSF.rows - normPSF.rows) / 2;
    normPSF.copyTo(paddedPSF(cv::Rect(cx, cy, normPSF.cols, normPSF.rows)));
    fftShift(paddedPSF); // IMPORTANT: initial PSF is placed at center, fftShift moves PSF to (0,0)

    // FFT of PSF
    cv::Mat psfDFT;
    cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);

    // Conjugate and |H|^2
    std::vector<cv::Mat> psfPlanes(2);
    cv::split(psfDFT, psfPlanes);

    cv::Mat psfMag2 = psfPlanes[0].mul(psfPlanes[0]) + psfPlanes[1].mul(psfPlanes[1]);
    double psfMaxVal;
    cv::minMaxLoc(psfMag2, nullptr, &psfMaxVal);
    psfMag2 /= static_cast<float>(psfMaxVal + 1e-6f);

    psfPlanes[1] *= -1;
    cv::Mat psfConj;
    cv::merge(psfPlanes, psfConj);

    // SNR map for smart SNR distribution to make image corners noise less visible
    cv::Mat snrMap(inputF.size(), CV_32F);
    cv::Point center(inputF.cols / 2, inputF.rows / 2);
    float maxDist = std::sqrt(center.x * center.x + center.y * center.y);
    float blurLength = std::sqrt(psf.cols * psf.cols + psf.rows * psf.rows);
    float minFactor = std::clamp(0.7f - 0.015f * blurLength, 0.3f, 0.7f);

    for (int y = 0; y < snrMap.rows; ++y)
        for (int x = 0; x < snrMap.cols; ++x) {
            float dx = x - center.x;
            float dy = y - center.y;
            float dist = std::sqrt(dx * dx + dy * dy) / maxDist;
            float weight = std::cos(dist * CV_PI / 2.0f);
            snrMap.at<float>(y, x) = minFactor + (1.0f - minFactor) * weight;
        }

    // Build Wiener denominator
    cv::Mat snrWeight = 1.0f / (snrMap * snr + 1e-6f);
    cv::Mat wienerDenom = psfMag2 + snrWeight;

    cv::threshold(wienerDenom, wienerDenom, 1e-5f, 1.0f, cv::THRESH_TOZERO);

    cv::Point psfCenter(psf.cols / 2, psf.rows / 2);
    cv::Point blurVec(psf.cols - 1 - psfCenter.x, psf.rows - 1 - psfCenter.y);

    // Slightly suppress blur direction using directional frequency mask
    cv::Mat freqSuppression(inputF.size(), CV_32F, 1.0f);
    float maxFreq = std::sqrt(center.x * center.x + center.y * center.y);

    for (int y = 0; y < inputF.rows; ++y)
        for (int x = 0; x < inputF.cols; ++x) {
            float dx = x - center.x;
            float dy = y - center.y;
            float dot = (dx * blurVec.x + dy * blurVec.y) / maxFreq;
            float angleFactor = 1.0f - 0.25f * std::abs(dot / maxFreq);
            freqSuppression.at<float>(y, x) = std::clamp(angleFactor, 0.7f, 1.0f);
        }

    wienerDenom /= freqSuppression;
    float baseEps = std::clamp(0.0001f * blurLength, 1e-4f, 1e-3f);
    cv::max(wienerDenom, baseEps, wienerDenom);

    // Split into channels
    std::vector<cv::Mat> inputChannels;
    if (inputF.channels() == 1) inputChannels.push_back(inputF);
    else cv::split(inputF, inputChannels);

    // Shared Hann window
    cv::Mat hann = createHannWindow2D(inputF.rows, inputF.cols);

    // Shared feathering mask
    cv::Mat mask(input.size(), CV_32F, 1.0f);
    int feather = std::min(60, std::min(input.cols, input.rows) / 10);
    cv::rectangle(mask, cv::Rect(0, 0, input.cols, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, input.rows - feather, input.cols, feather), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(0, 0, feather, input.rows), 0.0f, -1);
    cv::rectangle(mask, cv::Rect(input.cols - feather, 0, feather, input.rows), 0.0f, -1);
    cv::GaussianBlur(mask, mask, cv::Size(2 * feather + 1, 2 * feather + 1), feather);

    std::vector<cv::Mat> outputChannels(inputChannels.size());

    // Parallel deconvolution
    cv::parallel_for_(cv::Range(0, static_cast<int>(inputChannels.size())), [&](const cv::Range& range) {
        for (int i = range.start; i < range.end; ++i) {
            cv::Mat chWin = inputChannels[i].mul(hann);
            cv::Mat imgDFT;
            cv::dft(chWin, imgDFT, cv::DFT_COMPLEX_OUTPUT);

            cv::Mat filtered;
            cv::mulSpectrums(imgDFT, psfConj, filtered, 0);

            std::vector<cv::Mat> fPlanes(2);
            cv::split(filtered, fPlanes);

            cv::Mat safeMask = (wienerDenom > 1e-3f);
            fPlanes[0].setTo(0, ~safeMask);
            fPlanes[1].setTo(0, ~safeMask);

            fPlanes[0] /= wienerDenom;
            fPlanes[1] /= wienerDenom;
            cv::merge(fPlanes, filtered);

            cv::Mat restored;
            cv::idft(filtered, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

            restored /= hann + 1e-4f;

            restored = restored(cv::Rect(padX, padY, input.cols, input.rows));
            cv::Mat origCrop = inputChannels[i](cv::Rect(padX, padY, input.cols, input.rows));
            restored = restored.mul(mask) + origCrop.mul(1.0f - mask);

            cv::min(restored, 1.0f, restored);
            cv::max(restored, 0.0f, restored);

            outputChannels[i] = restored;
        }
    });

    // Merge channels and convert
    if (outputChannels.size() == 1)
        output = outputChannels[0];
    else
        cv::merge(outputChannels, output);

    output.convertTo(output, CV_8U, 255.0);
}

// EXPERIMENTAL
cv::Mat estimateGhostMask(const cv::Mat& input, const cv::Mat& psf, float blurThreshold = 0.1f) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    // Estimate blur direction from PSF
    cv::Moments m = cv::moments(psf, true);
    double cx = psf.cols / 2.0;
    double cy = psf.rows / 2.0;
    double dx = (m.m10 / m.m00) - cx;
    double dy = (m.m01 / m.m00) - cy;

    if (std::abs(dx) + std::abs(dy) < 1e-3) {
        dx = 1.0; dy = 0.0;
    }

    // Build directional filter (Sobel-like)
    cv::Point2f dir(dx, dy);
    float len = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    dir *= 1.0f / len;

    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);

    cv::Mat proj = gradX * dir.x + gradY * dir.y;
    cv::Mat energy = proj.mul(proj);

    // Smooth and threshold
    cv::GaussianBlur(energy, energy, cv::Size(11, 11), 5);
    double maxVal;
    cv::minMaxLoc(energy, nullptr, &maxVal);

    cv::Mat mask;
    cv::threshold(energy, mask, blurThreshold * maxVal, 1.0, cv::THRESH_BINARY);

    #ifdef DEBUG
        visualizeMatrix(mask, "ghosting_mask.png");
    #endif

    return mask;
}

cv::Mat estimateGhostMaskHF(const cv::Mat& deblurred) {
    cv::Mat gray, blurred, lap1, lap2, diff;
    cv::cvtColor(deblurred, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1/255.0f);

    cv::GaussianBlur(gray, blurred, cv::Size(0,0), 2.0);
    cv::Laplacian(gray, lap1, CV_32F, 3);
    cv::Laplacian(blurred, lap2, CV_32F, 3);
    diff = cv::abs(lap1 - lap2);

    cv::normalize(diff, diff, 0, 1, cv::NORM_MINMAX);
    cv::GaussianBlur(diff, diff, cv::Size(15,15), 5);
    
    #ifdef DEBUG
        visualizeMatrix(diff, "ghosting_mask_hf.png");
    #endif

    return diff;  // soft ghostiness map [0..1]
}

cv::Mat estimateGhostMaskDirectional(const cv::Mat& deblurred, const cv::Mat& psf) {
    cv::Mat gray; 
    cv::cvtColor(deblurred, gray, cv::COLOR_BGR2GRAY); 
    gray.convertTo(gray, CV_32F, 1 / 255.0f);

    cv::Mat gx, gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0), cv::Sobel(gray, gy, CV_32F, 0, 1);

    cv::Moments m = cv::moments(psf, true);
    double cx = psf.cols / 2.0, cy = psf.rows / 2.0;
    double dx = (m.m10 / m.m00) - cx, dy = (m.m01 / m.m00) - cy;

    if (fabs(dx) + fabs(dy) < 1e-3) dx=1, dy=0;

    cv::Point2f dir(dx, dy);
    dir *= (1 / std::sqrt(dx * dx + dy * dy));

    cv::Mat proj = gx * dir.x + gy * dir.y;
    cv::Mat energy = proj.mul(proj);
    cv::GaussianBlur(energy, energy, cv::Size(13,13), 4);
    cv::normalize(energy, energy, 0, 1, cv::NORM_MINMAX);

    return energy;
}

cv::Mat fuseGhostMasks(const cv::Mat& m1, const cv::Mat& m2) {
    cv::Mat fuse = cv::max(m1, m2);
    cv::GaussianBlur(fuse, fuse, cv::Size(15,15), 5);
    return fuse;
}

cv::Mat smoothGhostRegions(const cv::Mat& input, const cv::Mat& ghostMask, float strength = 0.5f) {
    cv::Mat inputF;
    input.convertTo(inputF, CV_32F, 1 / 255.0f);

    cv::Mat smooth;
    cv::edgePreservingFilter(inputF, smooth, cv::RECURS_FILTER, 60, 0.7f);

    std::vector<cv::Mat> inCh, smCh, outCh;
    cv::split(inputF, inCh);
    cv::split(smooth, smCh);

    cv::Mat ghostF;
    ghostMask.convertTo(ghostF, CV_32F, 1.0 / 255.0);

    for (int i = 0; i < inCh.size(); ++i) {
        cv::Mat w2, w1;

        // w2 = strength * ghostF
        cv::multiply(ghostF, strength, w2, 1.0, CV_32F);

        // w1 = 1.0 - w2
        cv::subtract(1.0f, w2, w1, cv::noArray(), CV_32F);

        cv::Mat a, b, blended;
        cv::multiply(inCh[i], w1, a, 1.0, CV_32F);
        cv::multiply(smCh[i], w2, b, 1.0, CV_32F);
        cv::add(a, b, blended, cv::noArray(), CV_32F);

        outCh.push_back(blended);
    }

    cv::Mat outF;
    cv::merge(outCh, outF);
    cv::min(outF, 1.0f, outF);
    cv::max(outF, 0.0f, outF);

    cv::Mat output;
    outF.convertTo(output, CV_8U, 255.0f);

    return output;
}

void Deblurrer::suppressGhosting(const cv::Mat& deblurred, const cv::Mat& psf, const cv::Mat& original, cv::Mat& output) {
    CV_Assert(deblurred.size() == original.size() && deblurred.type() == original.type());
    std::cout << "[Info] Start suppressing ghosting artifacts.\n";

    cv::Mat mHF = estimateGhostMaskHF(deblurred);
    cv::Mat mDir = estimateGhostMaskDirectional(deblurred, psf);
    cv::Mat ghostMask = fuseGhostMasks(mHF, mDir);

    cv::threshold(ghostMask, ghostMask, 0.2f, 1.0f, cv::THRESH_TOZERO);
    cv::normalize(ghostMask, ghostMask, 0, 1, cv::NORM_MINMAX);

    cv::Mat cleaned = smoothGhostRegions(deblurred, ghostMask, 0.7f);

    cv::Mat deblurF, origF, maskCh;
    deblurred.convertTo(deblurF, CV_32F, 1 / 255.0f);
    original.convertTo(origF, CV_32F, 1 / 255.0f);

    std::vector<cv::Mat> masks(deblurred.channels(), ghostMask);
    cv::merge(masks, maskCh);

    cv::Mat invMaskCh;
    cv::subtract(1.0f, maskCh, invMaskCh, cv::noArray(), CV_32F);

    cv::Mat cleanedF;
    cleaned.convertTo(cleanedF, CV_32F, 1.0 / 255.0);

    cv::Mat part1 = cleanedF.mul(maskCh);
    cv::Mat part2 = deblurF.mul(invMaskCh);
    cv::Mat outF = part1 + part2;

    cv::min(outF, 1.0f, outF);
    cv::max(outF, 0.0f, outF);
    outF.convertTo(output, CV_8U, 255.0f);

    std::cout << "[Info] Finish suppressing ghosting artifacts.\n";
}

void Deblurrer::denoiseImage(cv::Mat& image, float strength = 10.0f, float edgeStrength = 0.4f) {
    if (image.empty()) {
        std::cerr << "[Error] Denoise input image is empty." << std::endl;
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
        std::cerr << "[Error] Gamma correction input image is empty." << std::endl;
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
        std::cerr << "Failed to load image: " << inputImagePath << std::endl;
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

    cv::Mat psf;
    estimatePSF(blurLength, blurAngleRad, psf);

    cv::Mat deblurred;
    wienerDeconvolution(blurred, psf, deblurred, snr);

    cv::Mat output;
    suppressGhosting(deblurred, psf, blurred, output);

    if (config_.denoise) {
        if (!output.empty()) {
            std::cout << "[Info] Applying post-deconvolution denoising.\n";
            denoiseImage(output);
        } else {
            std::cerr << "[Warn] Output image is empty. Skipping denoising and recovery step.\n";
        }
    }

    cv::imwrite(outputImagePath, output);
    copyMetadata(inputImagePath, outputImagePath);

    std::cout << "Deblurred image saved to: " << outputImagePath << "\n";
}