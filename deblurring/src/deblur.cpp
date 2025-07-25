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
    std::uniform_real_distribution<> exposureDist(1.0 / 100.0, 1.0 / 10.0); // seconds

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

void Deblurrer::findPSF(int blurLengthPx, float yawDeg, cv::Mat& syntheticPSF) {
    if (blurLengthPx < 1) blurLengthPx = 1;

    int ksize = std::max(blurLengthPx * 2 + 1, 15);
    if (ksize % 2 == 0) ksize += 1;

    syntheticPSF = cv::Mat::zeros(ksize, ksize, CV_32F);
    cv::Point2f center(static_cast<float>(ksize) / 2.0f, static_cast<float>(ksize) / 2.0f);

    float angleRad = yawDeg * CV_PI / 180.0f;

    float dx = std::cos(angleRad);
    float dy = std::sin(angleRad);

    float halfLen = blurLengthPx * 0.5f;
    cv::Point2f pt1(center.x - dx * halfLen, center.y - dy * halfLen);
    cv::Point2f pt2(center.x + dx * halfLen, center.y + dy * halfLen);

    cv::line(syntheticPSF, pt1, pt2, cv::Scalar(1.0f), 1, cv::LINE_AA);

    int blurSize = (blurLengthPx > 20) ? 5 : 3;
    float sigma = (blurLengthPx > 20) ? 1.0f : 0.3f;
    if (blurSize % 2 == 0) blurSize += 1;
    cv::GaussianBlur(syntheticPSF, syntheticPSF, cv::Size(blurSize, blurSize), sigma, sigma);

    double sumVal = cv::sum(syntheticPSF)[0];
    if (sumVal <= 0.0) {
        std::cerr << "[ERROR] PSF generation failed — normalization invalid.\n";
        syntheticPSF.setTo(0);
        return;
    }
    syntheticPSF /= static_cast<float>(sumVal);

    cv::Mat psfDebug;
    cv::normalize(syntheticPSF, psfDebug, 0, 255, cv::NORM_MINMAX);
    psfDebug.convertTo(psfDebug, CV_8U);
    cv::imwrite("debug_psf.png", psfDebug);
}

void Deblurrer::blurImage(const std::string &inputPath, const std::string &outputImagePath, bool grayscale) {
    int imreadFlag = grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
    cv::Mat normal = cv::imread(inputPath, imreadFlag);
    if (normal.empty()) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return; 
    }

    std::unordered_map<std::string, std::string> metadata = extractImageMetadata(inputPath);
    if (!metadata.count("FlightSpeed")) {
        metadata = createSyntheticMetadata();
        assignMetadata(inputPath, metadata);
    }
    float speed = std::stof(metadata["FlightSpeed"]);
    float yaw = std::stof(metadata["DroneYaw"]);
    float exposure;
    std::string exposure_str = metadata["Exposure Time"];
    size_t slash_pos = exposure_str.find('/');

    // as exposure is usually written as 1/10, 1/1024, etc.
    if (slash_pos != std::string::npos) {
        float numerator = std::stof(exposure_str.substr(0, slash_pos));
        float denominator = std::stof(exposure_str.substr(slash_pos + 1));
        if (denominator != 0) {
            exposure = numerator / denominator;
        } else {
            std::cerr << "Warning: Exposure Time denominator is zero. Defaulting to 1.0s.\n";
            exposure = 1.0f;
        }
    } else {
        exposure = std::stof(exposure_str);
    }

    float gsd = calculateGSD(inputPath); // mm/px
    float blur = speed * 1000.0f * exposure; // mm
    int blurLength = static_cast<int>(blur / gsd); // px
    blurLength = std::max(1, blurLength);

    std::cout << "Original metadata: speed = " << speed << " m/s, yaw = " << yaw << " deg, exposure = " << exposure << " s (" << blurLength << " px blur)" << std::endl;    

    cv::Mat psf;
    findPSF(blurLength, yaw, psf);
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

void Deblurrer::createTestImage() {
    std::string imageSynthetic = "synthetic.jpg";
    std::string blurredImage = "synthetic_blurred.jpg";

    cv::Mat synthetic = createSyntheticTestImage();
    cv::imwrite(imageSynthetic, synthetic);

    std::unordered_map<std::string, std::string> metadata = createSyntheticMetadata();
    assignMetadata(imageSynthetic, metadata);

    blurImage(imageSynthetic, blurredImage, false);
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

void Deblurrer::wienerDeconvolution(const cv::Mat& input, const cv::Mat& psf, cv::Mat& output, float snr = 300.0f) {
    if (input.empty() || psf.empty()) {
        std::cerr << "[ERROR] Input or PSF is empty.\n";
        return;
    }

    // Convert to float32
    cv::Mat inputF;
    input.convertTo(inputF, CV_32F, 1.0 / 255.0);

    // Pad input with reflection to reduce edge artifacts
    int padY = psf.rows / 2;
    int padX = psf.cols / 2;
    cv::copyMakeBorder(inputF, inputF, padY, padY, padX, padX, cv::BORDER_REFLECT_101);

    // Normalize PSF
    cv::Mat normPSF;
    psf.convertTo(normPSF, CV_32F);
    double sumPSF = cv::sum(normPSF)[0];
    if (sumPSF <= 0.0) {
        std::cerr << "[ERROR] PSF sum is zero.\n";
        return;
    }
    normPSF /= static_cast<float>(sumPSF);

    // Pad PSF to match padded input size
    cv::Mat paddedPSF = cv::Mat::zeros(inputF.rows, inputF.cols, CV_32F);
    int x = (paddedPSF.cols - normPSF.cols) / 2;
    int y = (paddedPSF.rows - normPSF.rows) / 2;
    normPSF.copyTo(paddedPSF(cv::Rect(x, y, normPSF.cols, normPSF.rows)));
    fftShift(paddedPSF);  // Shift PSF center

    // DFT of PSF
    cv::Mat psfDFT;
    cv::dft(paddedPSF, psfDFT, cv::DFT_COMPLEX_OUTPUT);

    // Split image into channels
    std::vector<cv::Mat> channels;
    if (inputF.channels() == 1) {
        channels.push_back(inputF);
    } else {
        cv::split(inputF, channels);
    }

    std::vector<cv::Mat> resultChannels;
    for (auto& ch : channels) {
        cv::Mat hann = createHannWindow2D(ch.rows, ch.cols);
        cv::Mat chWindowed = ch.mul(hann);

        // Image DFT
        cv::Mat imgDFT;
        cv::dft(chWindowed, imgDFT, cv::DFT_COMPLEX_OUTPUT);

        // Split PSF
        std::vector<cv::Mat> psfPlanes(2);
        cv::split(psfDFT, psfPlanes);
        cv::Mat psfMag2;
        cv::magnitude(psfPlanes[0], psfPlanes[1], psfMag2);
        psfMag2 = psfMag2.mul(psfMag2) + (1.0f / snr);
        psfMag2 += 1e-6f;

        // Conjugate PSF
        psfPlanes[1] *= -1;
        cv::Mat psfConj;
        cv::merge(psfPlanes, psfConj);

        // Multiply image and PSF in frequency domain
        std::vector<cv::Mat> imgPlanes(2);
        cv::split(imgDFT, imgPlanes);
        cv::Mat real = imgPlanes[0].mul(psfPlanes[0]) - imgPlanes[1].mul(psfPlanes[1]);
        cv::Mat imag = imgPlanes[0].mul(psfPlanes[1]) + imgPlanes[1].mul(psfPlanes[0]);

        real /= psfMag2;
        imag /= psfMag2;

        cv::Mat wienerDFT;
        cv::merge(std::vector<cv::Mat>{ real, imag }, wienerDFT);

        // Inverse DFT
        cv::Mat restored;
        cv::idft(wienerDFT, restored, cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);

        // Avoid division by small Hann values
        cv::Mat safeHann = hann + 1e-4f;
        restored = restored / safeHann;

        // Crop to original size
        restored = restored(cv::Rect(padX, padY, input.cols, input.rows));
        cv::Mat origCropped = ch(cv::Rect(padX, padY, input.cols, input.rows));

        // Feather noisy corners using Gaussian mask
        cv::Mat mask = cv::Mat::ones(restored.size(), CV_32F);
        int feather = std::min(30, std::min(input.cols, input.rows) / 10);
        cv::rectangle(mask, cv::Rect(0, 0, input.cols, feather), 0.0f, -1);
        cv::rectangle(mask, cv::Rect(0, input.rows - feather, input.cols, feather), 0.0f, -1);
        cv::rectangle(mask, cv::Rect(0, 0, feather, input.rows), 0.0f, -1);
        cv::rectangle(mask, cv::Rect(input.cols - feather, 0, feather, input.rows), 0.0f, -1);
        cv::GaussianBlur(mask, mask, cv::Size(2 * feather + 1, 2 * feather + 1), feather);

        // Blend with original input to suppress noise in corners
        restored = restored.mul(mask) + origCropped.mul(1.0f - mask);

        // Clamp
        cv::threshold(restored, restored, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::threshold(restored, restored, 0.0, 0.0, cv::THRESH_TOZERO);

        resultChannels.push_back(restored);
    }

    cv::Mat resultF;
    if (resultChannels.size() == 1) {
        resultF = resultChannels[0];
    } else {
        cv::merge(resultChannels, resultF);
    }
    resultF.convertTo(output, CV_8U, 255.0);
}

// doesn't work as expected, needs refactoring
void Deblurrer::denoiseImage(cv::Mat& image, float strength = 30.0f, float edgeStrength = 0.4f) {
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

void Deblurrer::deblurImage(const std::string &inputPath, const std::string &outputImagePath, float snr = 500.0) {
    cv::Mat blurred = cv::imread(inputPath);
    if (blurred.empty()) {
        std::cerr << "Failed to load image: " << inputPath << "\n";
        return;
    }

    auto metadata = extractImageMetadata(inputPath);
    float speed = std::stof(metadata["FlightSpeed"]);
    float yaw = std::stof(metadata["DroneYaw"]);

    float exposure; // as exposure is usually 1/10, 1/1024, etc.
    std::string exposure_str = metadata["Exposure Time"];
    size_t slash_pos = exposure_str.find('/');
    if (slash_pos != std::string::npos) {
        float numerator = std::stof(exposure_str.substr(0, slash_pos));
        float denominator = std::stof(exposure_str.substr(slash_pos + 1));
        if (denominator != 0) {
            exposure = numerator / denominator;
        } else {
            std::cerr << "Warning: Exposure Time denominator is zero. Defaulting to 1.0s.\n";
            exposure = 1.0f;
        }
    } else {
        exposure = std::stof(exposure_str);
    }

    float gsd = calculateGSD(inputPath);
    float blur_mm = speed * 1000.0f * exposure;

    std::cout << "DEBUG: Calculated blur_mm: " << blur_mm << " mm\n";

    int blur_px = std::max(1, static_cast<int>(blur_mm / gsd));

    std::cout << "Deblurring with blur length: " << blur_px << " px, yaw: " << yaw << " deg\n";

    cv::Mat psf;
    findPSF(blur_px, yaw, psf);

    cv::Mat deblurred;
    wienerDeconvolution(blurred, psf, deblurred, snr);
    //cv::imwrite("temp_deblurred.jpg", deblurred); // needed only if denoising + other stuff

    // if (!deblurred.empty()) {
    //     std::cout << "[Info] Applying post-deconvolution denoising.\n";
    //     denoiseImage(deblurred);

    //     // float gamma = 0.7f;
    //     // recoverBrightness(deblurred, gamma);
    // } else {
    //     std::cerr << "[Warning] Output image is empty. Skipping denoising and recovery step.\n";
    // }

    cv::imwrite(outputImagePath, deblurred);
    copyMetadata(inputPath, outputImagePath);
    std::cout << "Deblurred image saved to: " << outputImagePath << "\n";
}

int main(int argc, char** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--generate-test") == 0) { // create synthetic image
        Deblurrer db;
        db.createTestImage();
        return 0;
    } else if (argc == 3 and std::strcmp(argv[1], "--blur") == 0) { // blur image instead of deblurring (useful for testing)
        std::string imageToBlur = argv[2];

        std::string imageName = imageToBlur.substr(0, imageToBlur.find('.'));
        std::string imageExtension = imageToBlur.substr(imageToBlur.find('.'), imageToBlur.size());
        std::string blurredImage = imageName + "_blurred" + imageExtension;

        Deblurrer deblurrer;
        deblurrer.blurImage(imageToBlur, blurredImage, false);
        return 0;
    } else if (argc == 2) { // deblurring logic
        std::string blurredImage = argv[1];

        std::string imageName = blurredImage.substr(0, blurredImage.find('.'));
        std::string imageExtension = blurredImage.substr(blurredImage.find('.'), blurredImage.size());
        std::string deblurredImage = imageName + "_deblurred" + imageExtension;

        Deblurrer deblurrer;
        deblurrer.deblurImage(blurredImage, deblurredImage);
        return 0;
    } else if (argc == 4 && std::strcmp(argv[2], "--snr") == 0) {
        std::string blurredImage = argv[1];
        float snr = std::stof(argv[3]);

        std::string imageName = blurredImage.substr(0, blurredImage.find('.'));
        std::string imageExtension = blurredImage.substr(blurredImage.find('.'), blurredImage.size());
        std::string deblurredImage = imageName + "_deblurred" + imageExtension;

        Deblurrer deblurrer;
        deblurrer.deblurImage(blurredImage, deblurredImage, snr);
    } else {
        std::cerr << "Wrong arguments.\n";
        return 1;
    }
}

