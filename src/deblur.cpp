#include <cstdio>
#include <ctime>
#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <thread>

namespace fs = std::filesystem;

constexpr const char* OUTPUT_DIR = "deblurred";

cv::Mat createInitialKernel(int size) {
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    kernel.row(size / 2).setTo(1.0f);
    kernel /= cv::sum(kernel)[0];
    return kernel;
}

cv::Mat convolve(const cv::Mat& image, const cv::Mat& kernel) {
    cv::Mat result;
    cv::filter2D(image, result, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
    return result;
}

cv::Mat richardsonLucy(const cv::Mat& image, const cv::Mat& kernel, int iterations) {
    cv::Mat estimate = cv::Mat::ones(image.size(), CV_32F);
    cv::Mat kernelFlipped;
    cv::flip(kernel, kernelFlipped, -1);

    for (int i = 0; i < iterations; ++i) {
        cv::Mat estConv;
        cv::filter2D(estimate, estConv, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
        cv::Mat ratio = image / (estConv + 1e-6f);
        cv::Mat corr;
        cv::filter2D(ratio, corr, -1, kernelFlipped, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
        estimate = estimate.mul(corr);
    }
    return estimate;
}

void blindDeblur(const cv::Mat& blurred, cv::Mat& latent, cv::Mat& kernel, int iterations = 10, int rl_iters = 20) {
    int ksize = kernel.rows;

    for (int iter = 0; iter < iterations; ++iter) {
        std::cout << "Iteration: " << iter+1 << "/" << iterations << std::endl;

        // estimate latent image using RL
        latent = richardsonLucy(blurred, kernel, rl_iters);

        // estimate new kernel
        cv::Mat conv = convolve(latent, kernel);
        cv::Mat residual;
        cv::subtract(blurred, conv, residual, cv::noArray(), CV_32F);  // force output type


        // find edges in latent image
        cv::Mat gradX, gradY, gradMag;
        cv::Sobel(latent, gradX, CV_32F, 1, 0);
        cv::Sobel(latent, gradY, CV_32F, 0, 1);
        cv::magnitude(gradX, gradY, gradMag);

        // make strong edge mask
        cv::Mat gradMag8U;
        gradMag.convertTo(gradMag8U, CV_8U, 255.0);  // scale float [0,1] to 8-bit [0,255]
        double thresh = cv::threshold(gradMag8U, gradMag8U, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        cv::Mat mask = gradMag8U > thresh;

        // kernel refinement (cross-correlation idea)
        cv::Mat newKernel = cv::Mat::zeros(kernel.size(), CV_32F);
        for (int y = 0; y < ksize; ++y) {
            for (int x = 0; x < ksize; ++x) {
                cv::Mat shifted;
                cv::Mat shiftedLatent = latent.clone();
                cv::Mat tmp;
                cv::copyMakeBorder(shiftedLatent, tmp, y, ksize-y, x, ksize-x, cv::BORDER_REFLECT);
                shifted = tmp(cv::Rect(x, y, latent.cols, latent.rows));
                cv::Mat maskFloat;
                mask.convertTo(maskFloat, CV_32F, 1.0 / 255.0);  // convert to float mask (0.0 or 1.0)

                cv::Mat prod = (shifted.mul(blurred)).mul(maskFloat);

                newKernel.at<float>(y, x) = static_cast<float>(cv::sum(prod)[0]);
            }
        }
        cv::normalize(newKernel, kernel, 1, 0, cv::NORM_L1);
    }
}

void deblurImage(const std::string& imagePath) {
    cv::Mat blurredColor = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (blurredColor.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return;
    }

    blurredColor.convertTo(blurredColor, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(blurredColor, channels);

    int kernel_size = 15;
    cv::Mat kernel = createInitialKernel(kernel_size);
    std::vector<cv::Mat> latentChannels(3);

    for (int i = 0; i < 3; ++i) {
        cv::Mat kernelCopy = createInitialKernel(kernel_size);  // optionally reinitialize per channel
        blindDeblur(channels[i], latentChannels[i], kernelCopy, 8, 20);
    }    

    cv::Mat latentColor;
    cv::merge(latentChannels, latentColor);

    cv::Mat output8U;
    latentColor.convertTo(output8U, CV_8UC3, 255.0);

    time_t timer = std::time(nullptr);
    fs::create_directories(OUTPUT_DIR);
    std::string filename = OUTPUT_DIR + std::string("/deblurred_") + std::to_string(timer) + ".png";
    cv::imwrite(filename, output8U);
    std::cout << "Saved: " << filename << std::endl;
}

void interactive() {
    while (true) {
        std::string path;
        std::cout << "Enter image path: ";
        std::getline(std::cin, path);
        if (path.empty()) break;
        if (!fs::exists(path)) {
            std::cout << "No image exists at path: " << path << "\n";
            continue;
        }
        deblurImage(path);
    }
}

void monitor(const std::string& dir) {
    std::cout << "Monitoring: " << dir << "\n";
    while (true) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;
            deblurImage(entry.path().string());
        }
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
}

int main(int argc, char** argv) {
    if (argc == 1) {
        if (!fs::exists(OUTPUT_DIR)) fs::create_directory(OUTPUT_DIR);
        interactive();
    } else if (argc == 3 && std::string(argv[1]) == "-a") {
        if (!fs::exists(OUTPUT_DIR)) fs::create_directory(OUTPUT_DIR);
        std::string watchDir = argv[2];
        monitor(watchDir);
    } else {
        std::cerr << "Wrong arguments.\n";
        return 1;
    }    
    return 0;
}
