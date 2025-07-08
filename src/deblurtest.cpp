#include <iostream>
#include <string>
#include <filesystem>
#include <exception>

#include "../include/deblur.h"
#include "../include/dblrutils.h"

constexpr const char* OUTPUT_DIR = "deblurred";

int main() {
    std::string imagePath;
    std::cout << "Enter path to blurred image: ";
    std::cin >> imagePath;

    cv::Mat original_blurred_color = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (original_blurred_color.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return -1;
    }

    cv::Mat original_blurred_gray;
    cv::cvtColor(original_blurred_color, original_blurred_gray, cv::COLOR_BGR2GRAY);
    original_blurred_gray.convertTo(original_blurred_gray, CV_32F, 1.0 / 255.0);

    int num_scales = 3;
    float scale_factor = 0.5f;
    int initial_kernel_size = 25;

    // Prepare the image for the coarsest scale
    cv::Mat blurred_current_scale = original_blurred_gray.clone();
    for (int i = 0; i < num_scales - 1; ++i) {
        cv::resize(blurred_current_scale, blurred_current_scale, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
    }

    // Initialize the deblurrer for the coarsest scale
    BlindDeblurrer deblurrer;
    deblurrer.setupForScale(blurred_current_scale, initial_kernel_size);

    std::cout << "\n--- Initial Setup Report ---" << std::endl;
    std::cout << "Original Image size: " << original_blurred_gray.cols << "x" << original_blurred_gray.rows << std::endl;
    std::cout << "Coarsest Scale Image size: " << deblurrer.current_img_size.width << "x" << deblurrer.current_img_size.height << std::endl;
    std::cout << "Initial Kernel size: " << deblurrer.f.cols << "x" << deblurrer.f.rows << std::endl;
    std::cout << "Optimal DFT size: " << deblurrer.current_dft_size.width << "x" << deblurrer.current_dft_size.height << std::endl;
    std::cout << "ADMM Rho: " << deblurrer.rho << ", Tau: " << deblurrer.tau << std::endl;
    std::cout << "Priors Lambda1: " << deblurrer.lambda1 << ", Lambda2: " << deblurrer.lambda2 << ", Sigma1: " << deblurrer.sigma1 << std::endl;

    // --- Test L-step setup components ---
    std::filesystem::create_directories(OUTPUT_DIR);

    // 1. M_mask: Should be a binary image (0 or 1) indicating smooth regions.
    // Save M_mask for visual inspection. Multiply by 255 to make it visible.
    cv::imwrite(std::string(OUTPUT_DIR) + "/M_mask_debug.png", deblurrer.M_mask * 255);
    std::cout << "\nDebug: M_mask saved to " << OUTPUT_DIR << "/M_mask_debug.png" << std::endl;

    // 2. F_deriv_filters: Check their types and dimensions. They should be complex (CV_32FC2).
    if (!deblurrer.F_deriv_filters.empty()) {
        std::cout << "Debug: F_deriv_filters[0] (Identity Filter FFT):" << std::endl;
        std::cout << "  Dimensions: " << deblurrer.F_deriv_filters[0].cols << "x" << deblurrer.F_deriv_filters[0].rows << std::endl;
        std::cout << "  Type: " << deblurrer.F_deriv_filters[0].type() << " (Expected CV_32FC2 = 13)" << std::endl;
        // You can check other filters similarly if needed
    } else {
        std::cout << "Debug: F_deriv_filters is empty." << std::endl;
    }

    std::cout << "\nInitial setup and L-step helper preparation complete. Ready to implement updateLStep." << std::endl;

    // --- Call and Test the L-step ---
    std::cout << "Executing one L-step iteration..." << std::endl;
    try {
        deblurrer.updateLStep();

        // Save the updated latent image L for inspection
        cv::Mat L_display;
        deblurrer.L.convertTo(L_display, CV_8U, 255.0); // Convert to 8-bit for saving
        cv::imwrite(std::string(OUTPUT_DIR) + "/L_after_1_Lstep.png", L_display);
        std::cout << "Updated L after 1 L-step saved to " << OUTPUT_DIR << "/L_after_1_Lstep.png" << std::endl;
        std::cout << "L-step completed successfully." << std::endl;

    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Exception during L-step: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception during L-step: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown Exception during L-step." << std::endl;
    }
    return 0;
}