#include <iostream>
#include <string>
#include <filesystem> // For std::filesystem::create_directories
#include <exception>
#include <limits>     // For std::numeric_limits

#include "../include/deblur.h"
#include "../include/dblrutils.h" // Assuming this contains your FFT/IFFT wrappers, shrink, etc.

constexpr const char* OUTPUT_DIR = "deblurred";

// Helper function to save intermediate L and f images
void saveIntermediateResults(const BlindDeblurrer& deblurrer, int iteration) {
    // Save L (latent image)
    cv::Mat L_display;
    deblurrer.L.convertTo(L_display, CV_8U, 255.0);
    cv::imwrite(std::string(OUTPUT_DIR) + "/L_iter_" + std::to_string(iteration) + ".png", L_display);

    // Save f (blur kernel) - shifted back to center for visualization
    cv::Mat f_to_save = deblurrer.f.clone(); // Work on a copy

    // Apply inverse dftShift to f_to_save to center the kernel for visualization
    // This assumes f's current state (deblurrer.f) has its peak at (0,0) due to previous dftShift logic
    // If your f is naturally centered, and the dftShift is only applied to current_padded_f,
    // then this inverse shift might not be needed or would need to be re-evaluated.
    // However, if the F-step output (f_updated_padded_real) is (0,0) centered,
    // and then you crop it directly to 'f', 'f' itself would be (0,0) centered.
    // Let's assume 'f' is stored in its "natural" (centered) form after performFStepIFFTAndPostProcess
    // and the dftShift is only applied when creating current_padded_f.
    // If so, you'd apply the shift *here* to visualize it centered.

    // If f is stored with its peak at (0,0) (due to the dftShift in performFStepIFFTAndPostProcess)
    // then to center it for display, you need to shift it back.
    // This is the inverse of the shift applied when creating current_padded_f.
    if (f_to_save.cols > 1 && f_to_save.rows > 1) {
        // Calculate the shift needed to move (0,0) back to (size/2, size/2)
        int cx = f_to_save.cols / 2;
        int cy = f_to_save.rows / 2;

        cv::Mat q0(f_to_save, cv::Rect(0, 0, cx, cy));         // Top-Left
        cv::Mat q1(f_to_save, cv::Rect(cx, 0, f_to_save.cols - cx, cy));    // Top-Right
        cv::Mat q2(f_to_save, cv::Rect(0, cy, cx, f_to_save.rows - cy));    // Bottom-Left
        cv::Mat q3(f_to_save, cv::Rect(cx, cy, f_to_save.cols - cx, f_to_save.rows - cy)); // Bottom-Right

        cv::Mat tmp;
        // Swap quadrants to shift DC component back to center
        q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
        q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    }
    
    cv::Mat f_display;
    double min_f, max_f;
    cv::minMaxLoc(f_to_save, &min_f, &max_f);

    if (max_f > 0) {
        f_to_save.convertTo(f_display, CV_8U, 255.0 / max_f);
    } else {
        f_display = cv::Mat::zeros(f_to_save.size(), CV_8U);
    }
    cv::imwrite(std::string(OUTPUT_DIR) + "/f_iter_" + std::to_string(iteration) + "_centered.png", f_display);
    
    std::cout << "Saved L and f for iteration " << iteration << " to " << OUTPUT_DIR << std::endl;
}

// Helper function to save final L and f images
void saveFinalResults(const BlindDeblurrer& deblurrer) {
    std::cout << "\n--- Saving Final Results ---" << std::endl;

    // Save Final Latent Image (L)
    cv::Mat L_final_display;
    deblurrer.L.convertTo(L_final_display, CV_8U, 255.0);
    cv::imwrite(std::string(OUTPUT_DIR) + "/L_final.png", L_final_display);
    std::cout << "Final latent image saved to " << OUTPUT_DIR << "/L_final.png" << std::endl;

    // Save Final Blur Kernel (f) - shifted back to center for visualization
    cv::Mat f_final_to_save = deblurrer.f.clone(); // Work on a copy

    // Apply inverse dftShift to f_final_to_save to center the kernel for visualization
    if (f_final_to_save.cols > 1 && f_final_to_save.rows > 1) {
        int cx = f_final_to_save.cols / 2;
        int cy = f_final_to_save.rows / 2;

        cv::Mat q0(f_final_to_save, cv::Rect(0, 0, cx, cy));
        cv::Mat q1(f_final_to_save, cv::Rect(cx, 0, f_final_to_save.cols - cx, cy));
        cv::Mat q2(f_final_to_save, cv::Rect(0, cy, cx, f_final_to_save.rows - cy));
        cv::Mat q3(f_final_to_save, cv::Rect(cx, cy, f_final_to_save.cols - cx, f_final_to_save.rows - cy));

        cv::Mat tmp;
        q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
        q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
    }

    cv::Mat f_final_display;
    double min_f_final, max_f_final;
    cv::minMaxLoc(f_final_to_save, &min_f_final, &max_f_final);
    if (max_f_final > 0) {
        f_final_to_save.convertTo(f_final_display, CV_8U, 255.0 / max_f_final);
    } else {
        f_final_display = cv::Mat::zeros(f_final_to_save.size(), CV_8U);
    }
    cv::imwrite(std::string(OUTPUT_DIR) + "/f_final_centered.png", f_final_display);
    std::cout << "Final blur kernel saved to " << OUTPUT_DIR << "/f_final_centered.png" << std::endl;
}


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

    cv::Mat blurred_current_scale = original_blurred_gray.clone();
    int initial_kernel_size = 25;

    BlindDeblurrer deblurrer;
    deblurrer.setupForScale(blurred_current_scale, initial_kernel_size);

    std::cout << "\n--- Initial Setup Report ---" << std::endl;
    std::cout << "Original Image size: " << original_blurred_gray.cols << "x" << original_blurred_gray.rows << std::endl;
    std::cout << "Coarsest Scale Image size: " << deblurrer.current_img_size.width << "x" << deblurrer.current_img_size.height << std::endl;
    std::cout << "Initial Kernel size: " << deblurrer.f.cols << "x" << deblurrer.f.rows << std::endl;
    std::cout << "Optimal DFT size: " << deblurrer.current_dft_size.width << "x" << deblurrer.current_dft_size.height << std::endl;
    std::cout << "ADMM Rho: " << deblurrer.rho << ", Tau: " << deblurrer.tau << std::endl;
    std::cout << "Priors Lambda1: " << deblurrer.lambda1 << ", Lambda2: " << deblurrer.lambda2 << ", Sigma1: " << deblurrer.sigma1 << std::endl;

    std::filesystem::create_directories(OUTPUT_DIR);

    cv::imwrite(std::string(OUTPUT_DIR) + "/M_mask_initial.png", deblurrer.M_mask * 255);
    std::cout << "Debug: M_mask saved to " << OUTPUT_DIR << "/M_mask_initial.png" << std::endl;

    int num_admm_iterations = 50;

    std::cout << "\n--- Starting ADMM Iterations (Total: " << num_admm_iterations << ") ---" << std::endl;

    for (int i = 0; i < num_admm_iterations; ++i) {
        std::cout << "\n--- ADMM Iteration " << i + 1 << " ---" << std::endl;
        try {
            deblurrer.updateLStep();
            std::cout << "L-step completed for iteration " << i + 1 << std::endl;

            deblurrer.updateFStep();
            std::cout << "F-step completed for iteration " << i + 1 << std::endl;

            deblurrer.updateAuxAndLagrange();
            std::cout << "Auxiliary and Lagrange multipliers updated for iteration " << i + 1 << "\n";

            if ((i + 1) % 5 == 0) saveIntermediateResults(deblurrer, i + 1);

        } catch (const cv::Exception& e) {
            std::cerr << "OpenCV Exception during ADMM iteration " << i + 1 << ": " << e.what() << std::endl;
            break;
        } catch (const std::exception& e) {
            std::cerr << "Standard Exception during ADMM iteration " << i + 1 << ": " << e.what() << std::endl;
            break;
        } catch (...) {
            std::cerr << "Unknown Exception during ADMM iteration " << i + 1 << "." << std::endl;
            break;
        }
    }
    std::cout << "\n--- ADMM Iterations Complete ---" << std::endl;

    // Call the new helper function for final results
    saveFinalResults(deblurrer);

    return 0;
}