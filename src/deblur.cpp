#include "../include/deblur.h"
#include "../include/dblrutils.h"
#include <limits>
#include <complex>
#include <string>


void BlindDeblurrer::setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size) {
    current_img_size = blurred_image_scale.size();

    if (initial_kernel_size % 2 == 0) initial_kernel_size++;

    L = blurred_image_scale.clone();
    f = createInitialImpulseKernel(initial_kernel_size);
    cv::Size kernel_actual_size = f.size();

    h_x = cv::Mat::zeros(current_img_size, CV_32F);
    h_y = cv::Mat::zeros(current_img_size, CV_32F);
    lam_x = cv::Mat::zeros(current_img_size, CV_32F);
    lam_y = cv::Mat::zeros(current_img_size, CV_32F);

    p_k.resize(6);
    psi_k.resize(6);
    for (int i = 0; i < 6; ++i) {
        p_k[i] = cv::Mat::zeros(kernel_actual_size, CV_32F);
        psi_k[i] = cv::Mat::zeros(kernel_actual_size, CV_32F);
    }

    h_x_prev = cv::Mat::zeros(current_img_size, CV_32F);
    h_y_prev = cv::Mat::zeros(current_img_size, CV_32F);
    p_k_prev.resize(6);
    for (int i = 0; i < 6; ++i) {
        p_k_prev[i] = cv::Mat::zeros(kernel_actual_size, CV_32F);
    }

    zeta_k.assign(6, 1.0f);

    int required_dft_height = current_img_size.height + initial_kernel_size - 1;
    int required_dft_width = current_img_size.width + initial_kernel_size - 1;
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
    
    cv::Rect roi_initial_f_padded(0, 0, f.cols, f.rows);
    f.copyTo(current_padded_f(roi_initial_f_padded));

    computeDerivativeFiltersFFTs();
    initializeSpatialDerivativeKernels();
    computeLocalSmoothnessMask(blurred_image_scale);
}

cv::Mat BlindDeblurrer::createInitialImpulseKernel(int size) {
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    if (size > 0) {
        kernel.at<float>(0, 0) = 1.0f;
    }
    return kernel;
}

void BlindDeblurrer::computeGradientX(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel_x = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::filter2D(input, output, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

void BlindDeblurrer::computeGradientY(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel_y = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(input, output, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

void BlindDeblurrer::padImageForDFT(const cv::Mat& input, cv::Mat& padded_output, cv::Size& dft_size) {
    dft_size = getOptimalDFTSize(input.size());
    cv::copyMakeBorder(input, padded_output, 0, dft_size.height - input.rows, 0, dft_size.width - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
}

void BlindDeblurrer::computeDerivativeFiltersFFTs() {
    F_deriv_filters.clear();
    F_deriv_filters.resize(6);

    // Helper to prepare kernel for DFT (shift center to (0,0))
    auto prepare_kernel_for_dft = [&](const cv::Mat& small_kernel, cv::Mat& padded_output_fft) {
        cv::Mat padded_kernel = cv::Mat::zeros(current_dft_size, CV_32F);
        
        // Manual shifting equivalent to ifftshift for small kernels
        // This moves the logical center of the small kernel to (0,0) of the padded matrix
        for (int r = 0; r < small_kernel.rows; ++r) {
            for (int c = 0; c < small_kernel.cols; ++c) {
                // Calculate target position in padded matrix, wrapping around
                int target_r = (r + current_dft_size.height - small_kernel.rows / 2) % current_dft_size.height;
                int target_c = (c + current_dft_size.width - small_kernel.cols / 2) % current_dft_size.width;
                padded_kernel.at<float>(target_r, target_c) = small_kernel.at<float>(r, c);
            }
        }
        fft2d(padded_kernel, padded_output_fft);
    };

    // Identity filter (already correct impulse at 0,0)
    cv::Mat I_filter_padded = cv::Mat::zeros(current_dft_size, CV_32F);
    I_filter_padded.at<float>(0, 0) = 1.0f;
    fft2d(I_filter_padded, F_deriv_filters[0]);

    // Dx filter: [-0.5, 0, 0.5]
    cv::Mat small_Dx_kernel = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    prepare_kernel_for_dft(small_Dx_kernel, F_deriv_filters[1]);

    // Dy filter: [-0.5; 0; 0.5]
    cv::Mat small_Dy_kernel = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    prepare_kernel_for_dft(small_Dy_kernel, F_deriv_filters[2]);

    // Dxx filter: [1, -2, 1]
    cv::Mat small_Dxx_kernel = (cv::Mat_<float>(1, 3) << 1, -2, 1);
    prepare_kernel_for_dft(small_Dxx_kernel, F_deriv_filters[3]);

    // Dyy filter
    cv::Mat small_Dyy_kernel = (cv::Mat_<float>(3, 1) << 1, -2, 1);
    prepare_kernel_for_dft(small_Dyy_kernel, F_deriv_filters[4]);

    // Dxy filter
    cv::Mat small_Dxy_kernel = (cv::Mat_<float>(3, 3) <<
        0.25, 0, -0.25,
        0,    0,  0,
        -0.25, 0,  0.25);
    prepare_kernel_for_dft(small_Dxy_kernel, F_deriv_filters[5]);
}

void BlindDeblurrer::computeLocalSmoothnessMask(const cv::Mat& blurred_img_scale) {
    cv::Mat mean, sq_mean;
    cv::boxFilter(blurred_img_scale, mean, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    cv::boxFilter(blurred_img_scale.mul(blurred_img_scale), sq_mean, -1, cv::Size(3, 3), cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
    
    cv::Mat std_dev;
    cv::sqrt(cv::max(sq_mean - mean.mul(mean), 0), std_dev);

    float threshold_M = 0.01f;
    M_mask = (std_dev < threshold_M);
    M_mask.convertTo(M_mask, CV_32F);
}

void BlindDeblurrer::initializeSpatialDerivativeKernels() {
    spatial_deriv_kernels.resize(6);
    spatial_deriv_kernels[0] = (cv::Mat_<float>(1, 1) << 1.0f); // Identity kernel (for k=0, D0*f = f)
    spatial_deriv_kernels[1] = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5); // Dx kernel
    spatial_deriv_kernels[2] = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5); // Dy kernel
    spatial_deriv_kernels[3] = (cv::Mat_<float>(1, 3) << 1, -2, 1); // Dxx kernel
    spatial_deriv_kernels[4] = (cv::Mat_<float>(3, 1) << 1, -2, 1); // Dyy kernel (assuming 3x1 for vertical second derivative)
    spatial_deriv_kernels[5] = (cv::Mat_<float>(3, 3) << 0.25, 0, -0.25, 0, 0, 0, -0.25, 0, 0.25); // Dxy kernel (approximate cross derivative)
}

// --- L-step ---
LStepFFTInputsContainer BlindDeblurrer::prepareLStepInputsAndFFTs() {
    LStepFFTInputsContainer inputs;

    // Allocate padded matrices for h_x_prime and h_y_prime once
    cv::Mat h_x_prime_padded = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Mat h_y_prime_padded = cv::Mat::zeros(current_dft_size, CV_32F);

    // Define ROI for image-sized Mats once
    cv::Rect roi_image(0, 0, current_img_size.width, current_img_size.height);

    // FIX: Explicitly evaluate MatExpr to a cv::Mat before copyTo
    cv::Mat h_x_prime_unpadded_eval = h_x - lam_x / rho;
    h_x_prime_unpadded_eval.copyTo(h_x_prime_padded(roi_image));

    cv::Mat h_y_prime_unpadded_eval = h_y - lam_y / rho;
    h_y_prime_unpadded_eval.copyTo(h_y_prime_padded(roi_image));

    // Prepare p_k_prime_padded and its FFTs
    inputs.F_p_k_prime.resize(6);
    
    // Define ROI for kernel-sized Mats once
    cv::Rect roi_kernel(0, 0, p_k[0].cols, p_k[0].rows);

    for (int k = 0; k < 6; ++k) {
        cv::Mat p_k_prime_unpadded_eval = p_k[k] - psi_k[k] / rho;
        
        cv::Mat p_k_prime_padded_temp = cv::Mat::zeros(current_dft_size, CV_32F);
        p_k_prime_unpadded_eval.copyTo(p_k_prime_padded_temp(roi_kernel));
        fft2d(p_k_prime_padded_temp, inputs.F_p_k_prime[k]);
    }

    // Perform FFTs for h_x_prime and h_y_prime
    fft2d(h_x_prime_padded, inputs.F_h_x_prime);
    fft2d(h_y_prime_padded, inputs.F_h_y_prime);

    fft2d(current_padded_blurred, inputs.F_I);
    fft2d(current_padded_f, inputs.F_f);

    cv::Mat F_f_planes[2];
    cv::split(inputs.F_f, F_f_planes); // F_f is CV_32FC2
    F_f_planes[1] = -F_f_planes[1]; // Negate imaginary part
    cv::merge(F_f_planes, 2, inputs.F_f_conj_64FC2); // Merge into a 32FC2 temporary
    inputs.F_f_conj_64FC2.convertTo(inputs.F_f_conj_64FC2, CV_64FC2);

    return inputs;
}

cv::Mat BlindDeblurrer::computeLNumerator(const LStepFFTInputsContainer& inputs) {
    cv::Mat F_I_64FC2;
    inputs.F_I.convertTo(F_I_64FC2, CV_64FC2);

    cv::Mat term1;
    cv::mulSpectrums(F_I_64FC2, inputs.F_f_conj_64FC2, term1, 0, false);

    cv::Mat F_Dx_conj, F_Dy_conj;
    cv::Mat F_Dx_planes_temp[2], F_Dy_planes_temp[2];

    cv::Mat F_deriv_filter_1_64FC2, F_deriv_filter_2_64FC2;
    this->F_deriv_filters[1].convertTo(F_deriv_filter_1_64FC2, CV_64FC2);
    this->F_deriv_filters[2].convertTo(F_deriv_filter_2_64FC2, CV_64FC2);

    cv::split(F_deriv_filter_1_64FC2, F_Dx_planes_temp); F_Dx_planes_temp[1] = -F_Dx_planes_temp[1]; cv::merge(F_Dx_planes_temp, 2, F_Dx_conj);
    cv::split(F_deriv_filter_2_64FC2, F_Dy_planes_temp); F_Dy_planes_temp[1] = -F_Dy_planes_temp[1]; cv::merge(F_Dy_planes_temp, 2, F_Dy_conj);

    cv::Mat F_h_x_prime_64FC2, F_h_y_prime_64FC2;
    inputs.F_h_x_prime.convertTo(F_h_x_prime_64FC2, CV_64FC2);
    inputs.F_h_y_prime.convertTo(F_h_y_prime_64FC2, CV_64FC2);
   
    cv::Mat term2_part_x, term2_part_y;
    cv::mulSpectrums(F_h_x_prime_64FC2, F_Dx_conj, term2_part_x, 0, false);
    cv::mulSpectrums(F_h_y_prime_64FC2, F_Dy_conj, term2_part_y, 0, false);
    cv::Mat term2 = (term2_part_x + term2_part_y) * rho;
    
    cv::Mat term3 = cv::Mat::zeros(current_dft_size, CV_64FC2);
    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);
        
        cv::Mat F_deriv_k_conj;
        cv::Mat F_deriv_k_planes_temp[2];
        cv::split(F_deriv_k_64FC2, F_deriv_k_planes_temp); F_deriv_k_planes_temp[1] = -F_deriv_k_planes_temp[1]; cv::merge(F_deriv_k_planes_temp, 2, F_deriv_k_conj);
       
        cv::Mat F_p_k_prime_64FC2;
        inputs.F_p_k_prime[k].convertTo(F_p_k_prime_64FC2, CV_64FC2);
        
        cv::Mat temp_term;
        cv::mulSpectrums(F_p_k_prime_64FC2, F_deriv_k_conj, temp_term, 0, false);
        term3 += temp_term;
    }
    term3 *= rho;

    cv::Mat numerator_fft = term1 + term2 + term3;

    double min_num_fft, max_num_fft;
    cv::minMaxLoc(numerator_fft, &min_num_fft, &max_num_fft);
    std::cout << "Debug: Numerator FFT (Complex) - Min Val: " << min_num_fft << ", Max Val: " << max_num_fft << std::endl;

    cv::Vec2d dc_num = numerator_fft.at<cv::Vec2d>(0, 0);
    std::cout << "Debug: Numerator FFT DC component (0,0): (" << dc_num[0] << ", " << dc_num[1] << ")" << "\n";

    return numerator_fft;
}

cv::Mat BlindDeblurrer::computeLDenominator(const cv::Mat& F_f) {
    cv::Mat F_f_mag_sq;
    cv::Mat F_f_planes_temp_64F[2];
    cv::Mat F_f_planes_32F[2];
    cv::split(F_f, F_f_planes_32F);
    F_f_planes_32F[0].convertTo(F_f_planes_temp_64F[0], CV_64F);
    F_f_planes_32F[1].convertTo(F_f_planes_temp_64F[1], CV_64F);
    cv::magnitude(F_f_planes_temp_64F[0], F_f_planes_temp_64F[1], F_f_mag_sq);
    F_f_mag_sq = F_f_mag_sq.mul(F_f_mag_sq);

    cv::Mat F_Dx_real, F_Dx_imag, F_Dx_mag_sq;
    cv::Mat F_Dy_real, F_Dy_imag, F_Dy_mag_sq;

    cv::Mat F_deriv_filter_1_64FC2, F_deriv_filter_2_64FC2;
    this->F_deriv_filters[1].convertTo(F_deriv_filter_1_64FC2, CV_64FC2);
    this->F_deriv_filters[2].convertTo(F_deriv_filter_2_64FC2, CV_64FC2);

    cv::Mat split_temp_dx[2];
    cv::split(F_deriv_filter_1_64FC2, split_temp_dx);
    F_Dx_real = split_temp_dx[0];
    F_Dx_imag = split_temp_dx[1];
    cv::magnitude(F_Dx_real, F_Dx_imag, F_Dx_mag_sq);
    F_Dx_mag_sq = F_Dx_mag_sq.mul(F_Dx_mag_sq);

    cv::Mat split_temp_dy[2]; // Will be CV_64F
    cv::split(F_deriv_filter_2_64FC2, split_temp_dy);
    F_Dy_real = split_temp_dy[0];
    F_Dy_imag = split_temp_dy[1];
    cv::magnitude(F_Dy_real, F_Dy_imag, F_Dy_mag_sq);
    F_Dy_mag_sq = F_Dy_mag_sq.mul(F_Dy_mag_sq);

    cv::Mat term2_den = (F_Dx_mag_sq + F_Dy_mag_sq) * rho;

    cv::Mat term3_den = cv::Mat::zeros(current_dft_size, CV_64F);
    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);

        cv::Mat F_deriv_k_real, F_deriv_k_imag, F_deriv_k_mag_sq;
        cv::Mat split_temp_k[2];
        cv::split(F_deriv_k_64FC2, split_temp_k);
        F_deriv_k_real = split_temp_k[0];
        F_deriv_k_imag = split_temp_k[1];
        cv::magnitude(F_deriv_k_real, F_deriv_k_imag, F_deriv_k_mag_sq);
        F_deriv_k_mag_sq = F_deriv_k_mag_sq.mul(F_deriv_k_mag_sq);
        term3_den += F_deriv_k_mag_sq;
    }
    term3_den *= rho;

    cv::Mat denominator_fft_real_before_max = F_f_mag_sq + term2_den + term3_den;

    cv::Mat denominator_fft_real;
    cv::max(denominator_fft_real_before_max, 1e-6, denominator_fft_real); // small constant for stability

    cv::Mat denom_planes[] = {denominator_fft_real, cv::Mat::zeros(current_dft_size, CV_64F)};

    cv::Mat denominator_fft_complex;
    cv::merge(denom_planes, 2, denominator_fft_complex);

    cv::Vec2d dc_denom = denominator_fft_complex.at<cv::Vec2d>(0, 0);
    std::cout << "Debug: Denominator FFT DC component (0,0): (" << dc_denom[0] << ", " << dc_denom[1] << ")" << std::endl;

    return denominator_fft_complex;
}

void BlindDeblurrer::performLStepIFFTAndPostProcess(const cv::Mat& F_L_updated) {
    cv::Mat F_L_updated_ifft;
    ifft2d(F_L_updated, F_L_updated_ifft);

    cv::Mat ifft_planes[2];
    cv::split(F_L_updated_ifft, ifft_planes);
    cv::Mat real_part = ifft_planes[0];  // This is real spatial-domain image

    double min_L_padded, max_L_padded;
    cv::minMaxLoc(real_part, &min_L_padded, &max_L_padded); // Use the real part for min/max
    std::cout << "Debug: L_updated_padded (After IFFT, before clipping) - Min Val: " << min_L_padded << ", Max Val: " << max_L_padded << std::endl;
    std::cout << "Debug: L_updated_padded (After IFFT) - Norm of real part: " << cv::norm(real_part) << std::endl;


    L = real_part(cv::Rect(0, 0, current_img_size.width, current_img_size.height)).clone();
    L.convertTo(L, CV_32F);

    // Debug L after cropping and before clipping
    double min_L_cropped, max_L_cropped;
    cv::minMaxLoc(L, &min_L_cropped, &max_L_cropped);
    std::cout << "DEBUG_LSTEP_POST: L (after crop, before clip) Min: " << min_L_cropped << ", Max: " << max_L_cropped << ", Norm: " << cv::norm(L) << std::endl;

    cv::threshold(L, L, 1.0f, 1.0f, cv::THRESH_TRUNC); // Clip values > 1.0 to 1.0
    cv::threshold(L, L, 0.0f, 0.0f, cv::THRESH_TOZERO); // Clip values < 0.0 to 0.0

    double min_L_clipped, max_L_clipped;
    cv::minMaxLoc(L, &min_L_clipped, &max_L_clipped);
    std::cout << "DEBUG_LSTEP_POST: L (after clipping) Min: " << min_L_clipped << ", Max: " << max_L_clipped << ", Norm: " << cv::norm(L) << std::endl;

    current_padded_L = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Rect roi_L_pad(0, 0, current_img_size.width, current_img_size.height);
    L.copyTo(current_padded_L(roi_L_pad));
}

void BlindDeblurrer::updateLStep() {
    std::cout << "Debug: Entering updateLStep()...\n";

    // Initial L state
    double min_L_before, max_L_before;
    cv::minMaxLoc(L, &min_L_before, &max_L_before);
    std::cout << "DEBUG_LSTEP: L (before update) Min: " << min_L_before << ", Max: " << max_L_before << ", Norm: " << cv::norm(L) << std::endl;

    LStepFFTInputsContainer inputs = prepareLStepInputsAndFFTs();
    
    // Debug inputs.F_f and inputs.F_I
    double min_F_f, max_F_f, min_F_I, max_F_I;
    cv::minMaxLoc(inputs.F_f, &min_F_f, &max_F_f);
    cv::minMaxLoc(inputs.F_I, &min_F_I, &max_F_I);
    std::cout << "DEBUG_LSTEP_FFT_INPUTS: F_f (kernel FFT) Min: " << min_F_f << ", Max: " << max_F_f << ", Norm: " << cv::norm(inputs.F_f) << std::endl;
    std::cout << "DEBUG_LSTEP_FFT_INPUTS: F_I (blurred image FFT) Min: " << min_F_I << ", Max: " << max_F_I << ", Norm: " << cv::norm(inputs.F_I) << std::endl;

    cv::Mat numerator_fft = computeLNumerator(inputs);
    cv::Mat denominator_fft_complex = computeLDenominator(inputs.F_f);

    // Debug numerator and denominator after computation
    double min_num, max_num, min_den, max_den;
    cv::minMaxLoc(numerator_fft, &min_num, &max_num);
    cv::minMaxLoc(denominator_fft_complex, &min_den, &max_den);
    std::cout << "DEBUG_LSTEP_NUM_DEN: Numerator FFT Min: " << min_num << ", Max: " << max_num << ", Norm: " << cv::norm(numerator_fft) << std::endl;
    std::cout << "DEBUG_LSTEP_NUM_DEN: Denominator FFT Min: " << min_den << ", Max: " << max_den << ", Norm: " << cv::norm(denominator_fft_complex) << std::endl;


    cv::Mat F_L_updated = divideComplex(numerator_fft, denominator_fft_complex);

    double min_F_L_updated, max_F_L_updated;
    cv::minMaxLoc(F_L_updated, &min_F_L_updated, &max_F_L_updated);
    std::cout << "Debug: F_L_updated (After division, before IFFT) - Min Val: " << min_F_L_updated << ", Max Val: " << max_F_L_updated << ", Norm: " << cv::norm(F_L_updated) << std::endl;
    
    // Check DC component of F_L_updated
    cv::Vec2d dc_fl_updated = F_L_updated.at<cv::Vec2d>(0, 0);
    std::cout << "DEBUG_LSTEP: F_L_updated DC component (0,0): (" << dc_fl_updated[0] << ", " << dc_fl_updated[1] << ")" << std::endl;

    performLStepIFFTAndPostProcess(F_L_updated);

    double min_L_after, max_L_after;
    cv::minMaxLoc(L, &min_L_after, &max_L_after);
    std::cout << "DEBUG_LSTEP: L (after update and clipping) Min: " << min_L_after << ", Max: " << max_L_after << ", Norm: " << cv::norm(L) << std::endl;
    std::cout << "DEBUG_LSTEP: norm(L - current_img_size_L_before): " << cv::norm(L - cv::Mat(L.size(), L.type(), cv::Scalar(min_L_before))) << std::endl; // Needs a copy of L before update to compare

    std::cout << "Debug: updateLStep() completed successfully." << std::endl;
}

// --- F-Step ---
FStepFFTInputsContainer BlindDeblurrer::prepareFStepInputsAndFFTs() {
    FStepFFTInputsContainer inputs;

    cv::Mat current_padded_L_64F;
    current_padded_L.convertTo(current_padded_L_64F, CV_64F);
    fft2d(current_padded_L_64F, inputs.F_L);

    cv::Mat F_L_planes_temp[2];
    cv::split(inputs.F_L, F_L_planes_temp);
    F_L_planes_temp[1] = -F_L_planes_temp[1];
    cv::merge(F_L_planes_temp, 2, inputs.F_L_conj);

    cv::Mat current_padded_blurred_64F;
    current_padded_blurred.convertTo(current_padded_blurred_64F, CV_64F);
    fft2d(current_padded_blurred_64F, inputs.F_I_64FC2);

    inputs.F_p_k.resize(6);
    for (int k = 0; k < 6; ++k) {
        cv::Mat p_k_converted;
        p_k[k].convertTo(p_k_converted, CV_64F);

        cv::Mat p_k_padded_64F = cv::Mat::zeros(current_dft_size, CV_64F);
        cv::Rect roi_p_k(0, 0, p_k[k].cols, p_k[k].rows);
        p_k_converted.copyTo(p_k_padded_64F(roi_p_k));

        fft2d(p_k_padded_64F, inputs.F_p_k[k]);
    }
    return inputs;
}

cv::Mat BlindDeblurrer::computeFNumerator(const FStepFFTInputsContainer& inputs) {
    // Term 1: F_L_conj * F_I (element-wise multiplication)
    cv::Mat term1_num;
    cv::mulSpectrums(inputs.F_I_64FC2, inputs.F_L, term1_num, 0, true); // F_I * conj(F_L)

    cv::Mat sum_of_Dk_Pk_terms = cv::Mat::zeros(current_dft_size, CV_64FC2);

    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);

        cv::Mat temp_term; // conj(F_Dk) * F_Pk
        cv::mulSpectrums(inputs.F_p_k[k], F_deriv_k_64FC2, temp_term, 0, true);
        sum_of_Dk_Pk_terms += temp_term;
    }

    cv::Mat numerator_fft = term1_num + sum_of_Dk_Pk_terms * rho;
    return numerator_fft;
}

cv::Mat BlindDeblurrer::computeFDenominator(const FStepFFTInputsContainer& inputs) {
    // Term 1: |F_L|^2
    cv::Mat F_L_mag_sq;
    cv::Mat F_L_real_part, F_L_imag_part;
    cv::Mat F_L_planes[2];
    cv::split(inputs.F_L, F_L_planes);
    F_L_real_part = F_L_planes[0];
    F_L_imag_part = F_L_planes[1];
    cv::magnitude(F_L_real_part, F_L_imag_part, F_L_mag_sq); // Magnitude is sqrt(re^2 + im^2)
    F_L_mag_sq = F_L_mag_sq.mul(F_L_mag_sq); // Square the magnitude to get |F_L|^2

    // Term 2: rho * SUM(|F_Dk|^2)
    cv::Mat sum_of_Dk_mag_sq = cv::Mat::zeros(current_dft_size, CV_64F);

    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);
        
        cv::Mat F_deriv_k_real, F_deriv_k_imag, F_deriv_k_mag_sq;
        cv::Mat split_temp_k[2];
        cv::split(F_deriv_k_64FC2, split_temp_k);
        F_deriv_k_real = split_temp_k[0];
        F_deriv_k_imag = split_temp_k[1];
        cv::magnitude(F_deriv_k_real, F_deriv_k_imag, F_deriv_k_mag_sq);
        F_deriv_k_mag_sq = F_deriv_k_mag_sq.mul(F_deriv_k_mag_sq); // |F_Dk|^2
        sum_of_Dk_mag_sq += F_deriv_k_mag_sq;
    }

    cv::Mat denominator_fft_real_before_max = F_L_mag_sq + sum_of_Dk_mag_sq * rho;

    cv::Mat denominator_fft_real;
    cv::max(denominator_fft_real_before_max, 1e-6, denominator_fft_real);

    cv::Mat denom_planes[] = {denominator_fft_real, cv::Mat::zeros(current_dft_size, CV_64F)};

    std::cout << "Debug: Before merge in computeFDenominator:\n";
    std::cout << "  denominator_fft_real size: " << denominator_fft_real.size() << ", depth: " << denominator_fft_real.depth() << std::endl;
    std::cout << "  Expected zero_mat size: " << current_dft_size << ", depth: " << CV_64F << std::endl;
    std::cout << "  Zero_mat type: " << cv::Mat::zeros(current_dft_size, CV_64F).type() << std::endl; // For cross-check    

    cv::Mat denominator_fft_complex;
    cv::merge(denom_planes, 2, denominator_fft_complex);
    

    return denominator_fft_complex;
}

void BlindDeblurrer::performFStepIFFTAndPostProcess(const cv::Mat& F_f_updated) {
    cv::Mat f_updated_padded_complex;
    ifft2d(F_f_updated, f_updated_padded_complex);

    cv::Mat planes[2];
    cv::split(f_updated_padded_complex, planes); // planes[0] = real, planes[1] = imaginary
    cv::Mat f_updated_padded_real = planes[0];

    cv::Rect kernel_roi(0, 0, f.cols, f.rows);
    cv::Mat f_cropped = f_updated_padded_real(kernel_roi).clone();

    f_cropped.convertTo(f, CV_32F);
    cv::threshold(f, f, 0.0f, 0.0f, cv::THRESH_TOZERO);

    cv::Scalar sum_val = cv::sum(f);
    if (sum_val[0] > std::numeric_limits<float>::epsilon()) {
        f /= sum_val[0];
    } else {
        f = createInitialImpulseKernel(f.rows);
        std::cerr << "Warning: Kernel sum was zero/too small after F-step, re-initializing to impulse.\n";
    }

    current_padded_f = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Rect roi_f_in_padded(0, 0, f.cols, f.rows);
    f.copyTo(current_padded_f(roi_f_in_padded));
}


void BlindDeblurrer::updateFStep() {
    std::cout << "Executing one f-step iteration...\n";

    FStepFFTInputsContainer inputs = prepareFStepInputsAndFFTs();

    cv::Mat numerator_f_fft = computeFNumerator(inputs);
    cv::Mat denominator_f_fft_complex = computeFDenominator(inputs);
    
    double min_f_denom_mag, max_f_denom_mag;
    cv::Mat denom_mag_only;
    
    cv::Mat denom_planes[2];
    cv::split(denominator_f_fft_complex, denom_planes); 
    
    cv::magnitude(denom_planes[0], denom_planes[1], denom_mag_only); // denom_planes[0] is real, denom_planes[1] is imag
    
    cv::minMaxLoc(denom_mag_only, &min_f_denom_mag, &max_f_denom_mag);

    std::cout << "Debug: F-step Denominator FFT (complex) - Min Mag: " << min_f_denom_mag
              << ", Max Mag: " << max_f_denom_mag << "\n";

    cv::Mat F_f_updated = divideComplex(numerator_f_fft, denominator_f_fft_complex);

    performFStepIFFTAndPostProcess(F_f_updated);

    std::cout << "Debug: updateFStep() completed successfully.\n";
}

// --- Parameters update ---
cv::Mat BlindDeblurrer::shrink(const cv::Mat& z, float tau) {
    cv::Mat abs_z, sign_z, thresholded;
    cv::absdiff(z, cv::Scalar(0), abs_z);

    // Create thresholded magnitude
    cv::Mat abs_minus_tau = abs_z - tau;
    cv::max(abs_minus_tau, 0, thresholded);

    // Compute sign (vectorized)
    sign_z = cv::Mat::zeros(z.size(), z.type());
    sign_z.setTo(1.0f, z > 0);
    sign_z.setTo(-1.0f, z < 0);

    return sign_z.mul(thresholded);
}

void BlindDeblurrer::updateRho(const cv::Mat& L_x_current, const cv::Mat& L_y_current, const std::vector<cv::Mat>& f_Dk_current) {
    // float rho_factor = 1.1f; // Comment out or remove
    // float mu_balance = 10.0f; // Comment out or remove

    double r_norm_sq = 0.0;
    // ... (rest of r_norm_sq calculation) ...
    double r_norm = std::sqrt(r_norm_sq);

    double s_norm_sq = 0.0;
    // ... (rest of s_norm_sq calculation) ...
    double s_norm = rho * std::sqrt(s_norm_sq);

    std::cout << "Debug: r_norm: " << r_norm << ", s_norm: " << s_norm << std::endl;

    // // COMMENT OUT OR REMOVE THIS ENTIRE IF/ELSE BLOCK FOR FIXED RHO DEBUGGING
    // if (r_norm > mu_balance * s_norm) {
    //     rho *= rho_factor;
    //     std::cout << "Debug: Increasing rho. New rho: " << rho << std::endl;
    // } else if (s_norm > mu_balance * r_norm) {
    //     rho /= rho_factor;
    //     std::cout << "Debug: Decreasing rho. New rho: " << rho << std::endl;
    // } else {
    //     std::cout << "Debug: Rho unchanged. Rho: " << rho << std::endl;
    // }

    // float rho_min_val = 1e-3f;
    // rho = std::max(rho_min_val, std::min(rho_max, rho));
    
    // TEMPORARY: FIX RHO FOR DEBUGGING
    rho = 0.1f; // Or 0.01f, or 0.001f
    std::cout << "Debug: Rho fixed for debugging. Current Rho: " << rho << std::endl;

    std::cout << "Current new Rho: " + std::to_string(rho) + "\n";
}

void BlindDeblurrer::updateAuxAndLagrange() {
    std::cout << "Debug: Entering updateAuxAndLagrange()...\n";

    // Store previous values for s_norm calculation
    h_x.copyTo(h_x_prev);
    h_y.copyTo(h_y_prev);
    for (int k = 0; k < 6; ++k) {
        p_k[k].copyTo(p_k_prev[k]);
    }

    cv::Mat L_x, L_y;
    computeGradientX(L, L_x);
    computeGradientY(L, L_y);

    std::cout << "DEBUG_AUX_L_GRAD: L_x norm: " << cv::norm(L_x) << ", L_y norm: " << cv::norm(L_y) << std::endl;
    double min_Lx, max_Lx, min_Ly, max_Ly;
    cv::minMaxLoc(L_x, &min_Lx, &max_Lx);
    cv::minMaxLoc(L_y, &min_Ly, &max_Ly);
    std::cout << "DEBUG_AUX_L_GRAD: L_x min/max: " << min_Lx << "/" << max_Lx << ", L_y min/max: " << min_Ly << "/" << max_Ly << std::endl;

    // Update h_x
    cv::Mat term_h_x = L_x + lam_x / rho;
    float tau_hx = lambda1 / rho;
    std::cout << "DEBUG_AUX_HX: lambda1: " << lambda1 << ", rho: " << rho << ", tau_hx: " << tau_hx << std::endl;
    h_x = shrink(term_h_x, tau_hx);
    std::cout << "DEBUG_AUX_HX: h_x norm (after shrink): " << cv::norm(h_x) << std::endl;
    std::cout << "DEBUG_AUX_HX: h_x_prev norm: " << cv::norm(h_x_prev) << std::endl;
    std::cout << "DEBUG_AUX_HX: norm(h_x - h_x_prev): " << cv::norm(h_x - h_x_prev) << std::endl;

    // Update h_y
    cv::Mat term_h_y = L_y + lam_y / rho;
    float tau_hy = lambda1 / rho; // Should be same as tau_hx
    std::cout << "DEBUG_AUX_HY: lambda1: " << lambda1 << ", rho: " << rho << ", tau_hy: " << tau_hy << std::endl;
    h_y = shrink(term_h_y, tau_hy);
    std::cout << "DEBUG_AUX_HY: h_y norm (after shrink): " << cv::norm(h_y) << std::endl;
    std::cout << "DEBUG_AUX_HY: h_y_prev norm: " << cv::norm(h_y_prev) << std::endl;
    std::cout << "DEBUG_AUX_HY: norm(h_y - h_y_prev): " << cv::norm(h_y - h_y_prev) << std::endl;


    // Update Lagrange Multipliers for Image Gradients
    lam_x = lam_x + rho * (L_x - h_x);
    lam_y = lam_y + rho * (L_y - h_y);
    std::cout << "DEBUG_AUX_LAMBDA: lam_x norm: " << cv::norm(lam_x) << ", lam_y norm: " << cv::norm(lam_y) << std::endl;


    // Update p_k (Auxiliary Variables for Kernel Gradients)
    std::vector<cv::Mat> f_Dk_rho(6); // To store Dk*f for r_norm calculation
    for (int k = 0; k < 6; ++k) {
        cv::Mat f_Dk;
        cv::filter2D(f, f_Dk, -1, spatial_deriv_kernels[k], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        
        // Store for r_norm calculation
        f_Dk.copyTo(f_Dk_rho[k]);

        // Debug f_Dk norms
        double min_fDk, max_fDk;
        cv::minMaxLoc(f_Dk, &min_fDk, &max_fDk);
        std::cout << "DEBUG_AUX_PK[" << k << "]: f_Dk norm: " << cv::norm(f_Dk) << ", min/max: " << min_fDk << "/" << max_fDk << std::endl;

        cv::Mat term_p_k = f_Dk + psi_k[k] / rho;
        float tau_pk = lambda2 / rho / zeta_k[k];
        std::cout << "DEBUG_AUX_PK[" << k << "]: lambda2: " << lambda2 << ", rho: " << rho << ", zeta_k[" << k << "]: " << zeta_k[k] << ", tau_pk: " << tau_pk << std::endl;
        p_k[k] = shrink(term_p_k, tau_pk);
        std::cout << "DEBUG_AUX_PK[" << k << "]: p_k norm (after shrink): " << cv::norm(p_k[k]) << std::endl;
        std::cout << "DEBUG_AUX_PK[" << k << "]: p_k_prev[" << k << "] norm: " << cv::norm(p_k_prev[k]) << std::endl;
        std::cout << "DEBUG_AUX_PK[" << k << "]: norm(p_k[" << k << "] - p_k_prev[" << k << "]): " << cv::norm(p_k[k] - p_k_prev[k]) << std::endl;

        // Update Lagrange Multipliers for Kernel Gradients
        psi_k[k] = psi_k[k] + rho * (f_Dk - p_k[k]);
        std::cout << "DEBUG_AUX_PSI[" << k << "]: psi_k norm: " << cv::norm(psi_k[k]) << std::endl;
    }

    // Update Rho (Adaptive Parameter)
    updateRho(L_x, L_y, f_Dk_rho);
    std::cout << "ADMM Rho: " << rho << ", Effective Lambda1 Tau: " << lambda1 / rho << std::endl;
    std::cout << "Debug: Exiting updateAuxAndLagrange() successfully.\n";
}