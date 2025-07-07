#include <opencv4/opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <filesystem> 

constexpr const char* OUTPUT_DIR = "deblurred";

cv::Size getOptimalDFTSize(const cv::Size& size) {
    int r = cv::getOptimalDFTSize(size.height);
    int c = cv::getOptimalDFTSize(size.width);
    return cv::Size(c, r);
}

void fft2d(const cv::Mat& input_real, cv::Mat& output_complex) {
    cv::Mat planes[] = {input_real, cv::Mat::zeros(input_real.size(), CV_32F)};
    cv::merge(planes, 2, output_complex);
    cv::dft(output_complex, output_complex, cv::DFT_COMPLEX_OUTPUT);
}

void ifft2d(const cv::Mat& input_complex, cv::Mat& output_real) {
    cv::dft(input_complex, output_real, cv::DFT_INVERSE | cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
}

class BlindDeblurrer {
public:
    cv::Mat L; // latent image estimate
    cv::Mat f; // blur kernel estimate

    // ADMM auxiliary vars
    cv::Mat h_x, h_y; // variables for L's gradients
    cv::Mat lam_x, lam_y; // Lagrange multipliers for L's gradients
    std::vector<cv::Mat> p_k; // variables for noise derivatives
    std::vector<cv::Mat> psi_k; // Lagrange multipliers for noise derivatives

    // noise parameters (zeta_k for noise derivatives)
    std::vector<float> zeta_k;

    // ADMM and prior weights
    float rho;
    float tau;
    float lambda1;
    float lambda2;
    float sigma1;

    // current image and DFT dimensions
    cv::Size current_img_size;
    cv::Size current_dft_size;

    // padded versions for FFT operations
    cv::Mat current_padded_blurred;
    cv::Mat current_padded_L;
    cv::Mat current_padded_f;

    BlindDeblurrer() {
        rho = 0.1f;
        tau = 0.001f;
        lambda1 = 1.0f;
        lambda2 = 1.0f;
        sigma1 = 0.02f; // Initial standard deviation for local prior
    }

    void setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size) {
        current_img_size = blurred_image_scale.size();

        // odd kernel size
        if (initial_kernel_size % 2 == 0) initial_kernel_size++;

        L = blurred_image_scale.clone();
        f = createInitialImpulseKernel(initial_kernel_size);

        h_x = cv::Mat::zeros(current_img_size, CV_32F);
        h_y = cv::Mat::zeros(current_img_size, CV_32F);
        lam_x = cv::Mat::zeros(current_img_size, CV_32F);
        lam_y = cv::Mat::zeros(current_img_size, CV_32F);

        p_k.resize(6); // for 6 derivative types (identity, dx, dy, dxx, dxy, dyy)
        psi_k.resize(6);
        for (int i = 0; i < 6; ++i) {
            p_k[i] = cv::Mat::zeros(current_img_size, CV_32F);
            psi_k[i] = cv::Mat::zeros(current_img_size, CV_32F);
        }

        zeta_k.assign(6, 0.01f);

        int required_dft_height = current_img_size.height + initial_kernel_size - 1;
        int required_dft_width = current_img_size.width + initial_kernel_size - 1;
        // try to find a number which is a product of a small prime factors for performance,
        // e.g, 300 = 2^2 * 3 * 5^2

        current_dft_size = getOptimalDFTSize(cv::Size(required_dft_width, required_dft_height));

        cv::copyMakeBorder(blurred_image_scale, current_padded_blurred,
                           0, current_dft_size.height - current_img_size.height,
                           0, current_dft_size.width - current_img_size.width,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        cv::copyMakeBorder(L, current_padded_L,
                           0, current_dft_size.height - current_img_size.height,
                           0, current_dft_size.width - current_img_size.width,
                           cv::BORDER_CONSTANT, cv::Scalar::all(0));

        current_padded_f = cv::Mat::zeros(current_dft_size, CV_32F);
        cv::Rect roi(0, 0, initial_kernel_size, initial_kernel_size);
        f.copyTo(current_padded_f(roi));
    }

private:
    static cv::Mat createInitialImpulseKernel(int size) {
        cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
        if (size > 0) {
            kernel.at<float>(size / 2, size / 2) = 1.0f;
        }
        return kernel;
    }
};

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
    float scale_factor = 0.5f; // downsampling factor for each coarser scale
    int initial_kernel_size = 25; // start kernel guess for the coarsest scale

    cv::Mat blurred_current_scale = original_blurred_gray.clone();
    for (int i = 0; i < num_scales - 1; ++i) {
        cv::resize(blurred_current_scale, blurred_current_scale, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
    }

    // initialize the deblurrer for the coarsest scale
    BlindDeblurrer deblurrer;
    deblurrer.setupForScale(blurred_current_scale, initial_kernel_size);

    std::cout << "\n--- Initial Setup Report ---" << std::endl;
    std::cout << "Original Image size: " << original_blurred_gray.cols << "x" << original_blurred_gray.rows << std::endl;
    std::cout << "Coarsest Scale Image size: " << deblurrer.current_img_size.width << "x" << deblurrer.current_img_size.height << std::endl;
    std::cout << "Initial Kernel size: " << deblurrer.f.cols << "x" << deblurrer.f.rows << std::endl;
    std::cout << "Optimal DFT size: " << deblurrer.current_dft_size.width << "x" << deblurrer.current_dft_size.height << std::endl;
    std::cout << "ADMM Rho: " << deblurrer.rho << ", Tau: " << deblurrer.tau << std::endl;
    std::cout << "Priors Lambda1: " << deblurrer.lambda1 << ", Lambda2: " << deblurrer.lambda2 << ", Sigma1: " << deblurrer.sigma1 << std::endl;

    std::cout << "\nReady for ADMM iterations." << std::endl;

    return 0;
}