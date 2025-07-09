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

    zeta_k.assign(6, 0.01f);

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
    
    // Create a temporary shifted version of the initial kernel 'f'
    cv::Mat initial_f_shifted = cv::Mat::zeros(f.size(), CV_32F);
    int cx_init = f.cols / 2;
    int cy_init = f.rows / 2;

    cv::Mat q0_init(f, cv::Rect(0, 0, cx_init, cy_init));
    cv::Mat q1_init(f, cv::Rect(cx_init, 0, f.cols - cx_init, cy_init));
    cv::Mat q2_init(f, cv::Rect(0, cy_init, cx_init, f.rows - cy_init));
    cv::Mat q3_init(f, cv::Rect(cx_init, cy_init, f.cols - cx_init, f.rows - cy_init));

    q0_init.copyTo(initial_f_shifted(cv::Rect(f.cols - cx_init, f.rows - cy_init, cx_init, cy_init)));
    q1_init.copyTo(initial_f_shifted(cv::Rect(0, f.rows - cy_init, f.cols - cx_init, cy_init)));
    q2_init.copyTo(initial_f_shifted(cv::Rect(f.cols - cx_init, 0, cx_init, f.rows - cy_init)));
    q3_init.copyTo(initial_f_shifted(cv::Rect(0, 0, f.cols - cx_init, f.rows - cy_init)));

    // Copy the shifted initial kernel to the padded matrix
    cv::Rect roi_initial_f_padded(0, 0, f.cols, f.rows);
    initial_f_shifted.copyTo(current_padded_f(roi_initial_f_padded));

    computeDerivativeFiltersFFTs();
    initializeSpatialDerivativeKernels();
    computeLocalSmoothnessMask(blurred_image_scale);
}

cv::Mat BlindDeblurrer::createInitialImpulseKernel(int size) {
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_32F);
    if (size > 0) {
        kernel.at<float>(size / 2, size / 2) = 1.0f;
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

    // Identity filter
    cv::Mat I_filter_padded = cv::Mat::zeros(current_dft_size, CV_32F);
    I_filter_padded.at<float>(0, 0) = 1.0f;
    fft2d(I_filter_padded, F_deriv_filters[0]);

    // Dx filter: [-0.5, 0, 0.5]
    cv::Mat small_Dx_kernel = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::Mat padded_Dx_filter = cv::Mat::zeros(current_dft_size, CV_32F);
    // Copy the small kernel into the top-left corner of the padded matrix
    small_Dx_kernel.copyTo(padded_Dx_filter(cv::Rect(0, 0, small_Dx_kernel.cols, small_Dx_kernel.rows)));
    fft2d(padded_Dx_filter, F_deriv_filters[1]);

    // Dy filter: [-0.5; 0; 0.5]
    cv::Mat small_Dy_kernel = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::Mat padded_Dy_filter = cv::Mat::zeros(current_dft_size, CV_32F);
    small_Dy_kernel.copyTo(padded_Dy_filter(cv::Rect(0, 0, small_Dy_kernel.cols, small_Dy_kernel.rows)));
    fft2d(padded_Dy_filter, F_deriv_filters[2]);

    // Dxx filter: [1, -2, 1]
    cv::Mat small_Dxx_kernel = (cv::Mat_<float>(1, 3) << 1, -2, 1);
    cv::Mat padded_Dxx_filter = cv::Mat::zeros(current_dft_size, CV_32F);
    small_Dxx_kernel.copyTo(padded_Dxx_filter(cv::Rect(0, 0, small_Dxx_kernel.cols, small_Dxx_kernel.rows)));
    fft2d(padded_Dxx_filter, F_deriv_filters[3]);

    // Dxy filter: 2x2 filter scaled to 3x3 for central pixel alignment
    // Original: 0.25 -0.25
    //          -0.25  0.25
    // Using a 3x3 for central difference approximation
    cv::Mat small_Dxy_kernel = (cv::Mat_<float>(3, 3) <<
        0.25, 0, -0.25,
        0,    0,  0,
        -0.25, 0,  0.25);
    cv::Mat padded_Dxy_filter = cv::Mat::zeros(current_dft_size, CV_32F);
    small_Dxy_kernel.copyTo(padded_Dxy_filter(cv::Rect(0, 0, small_Dxy_kernel.cols, small_Dxy_kernel.rows)));
    fft2d(padded_Dxy_filter, F_deriv_filters[4]);

    // Dyy filter: [1; -2; 1]
    cv::Mat small_Dyy_kernel = (cv::Mat_<float>(3, 1) << 1, -2, 1);
    cv::Mat padded_Dyy_filter = cv::Mat::zeros(current_dft_size, CV_32F);
    small_Dyy_kernel.copyTo(padded_Dyy_filter(cv::Rect(0, 0, small_Dyy_kernel.cols, small_Dyy_kernel.rows)));
    fft2d(padded_Dyy_filter, F_deriv_filters[5]);
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
        // FIX: Explicitly evaluate MatExpr to a cv::Mat before copyTo
        cv::Mat p_k_prime_unpadded_eval = p_k[k] - psi_k[k] / rho;
        
        cv::Mat p_k_prime_padded_temp = cv::Mat::zeros(current_dft_size, CV_32F);
        p_k_prime_unpadded_eval.copyTo(p_k_prime_padded_temp(roi_kernel));
        
        // Perform FFT directly into inputs.F_p_k_prime[k]
        fft2d(p_k_prime_padded_temp, inputs.F_p_k_prime[k]);
    }

    // Perform FFTs for h_x_prime and h_y_prime
    fft2d(h_x_prime_padded, inputs.F_h_x_prime);
    fft2d(h_y_prime_padded, inputs.F_h_y_prime);

    // FFTs for blurred image and kernel (should already be CV_32F)
    fft2d(current_padded_blurred, inputs.F_I);
    fft2d(current_padded_f, inputs.F_f);

    // Compute F_f_conj using mulSpectrums
    cv::Mat F_f_conj_32FC2;
    // The `true` argument calculates the conjugate of the second operand (inputs.F_f)
    cv::mulSpectrums(inputs.F_f, inputs.F_f, F_f_conj_32FC2, 0, true);

    // Convert to CV_64FC2 if inputs.F_f_conj_64FC2 specifically needs this precision
    F_f_conj_32FC2.convertTo(inputs.F_f_conj_64FC2, CV_64FC2);

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
    cv::Mat L_updated_padded;
    ifft2d(F_L_updated, L_updated_padded);

    double min_L_padded, max_L_padded;
    cv::minMaxLoc(L_updated_padded, &min_L_padded, &max_L_padded);
    std::cout << "Debug: L_updated_padded (After IFFT, before clipping) - Min Val: " << min_L_padded << ", Max Val: " << max_L_padded << std::endl;

    L = L_updated_padded(cv::Rect(0, 0, current_img_size.width, current_img_size.height)).clone();
    L.convertTo(L, CV_32F);

    cv::threshold(L, L, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(L, L, 0.0f, 0.0f, cv::THRESH_TOZERO);

    current_padded_L = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Rect roi_L_pad(0, 0, current_img_size.width, current_img_size.height);
    L.copyTo(current_padded_L(roi_L_pad));
}

void BlindDeblurrer::updateLStep() {
    LStepFFTInputsContainer inputs = prepareLStepInputsAndFFTs();
    cv::Mat numerator_fft = computeLNumerator(inputs);
    cv::Mat denominator_fft_complex = computeLDenominator(inputs.F_f);

    cv::Mat F_L_updated = divideComplex(numerator_fft, denominator_fft_complex);

    double min_F_L_updated, max_F_L_updated;
    cv::minMaxLoc(F_L_updated, &min_F_L_updated, &max_F_L_updated);
    std::cout << "Debug: F_L_updated (After division, before IFFT) - Min Val: " << min_F_L_updated << ", Max Val: " << max_F_L_updated << std::endl;

    performLStepIFFTAndPostProcess(F_L_updated);
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
        cv::Mat p_k_padded_64F = cv::Mat::zeros(current_dft_size, CV_64F);
        cv::Rect roi_p_k(0, 0, p_k[k].cols, p_k[k].rows);
        p_k[k].convertTo(p_k_padded_64F(roi_p_k), CV_64F);

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
    cv::Mat f_updated_padded_real;
    ifft2d(F_f_updated, f_updated_padded_real);

    cv::Rect kernel_roi(0, 0, f.cols, f.rows); // This is needed for f_cropped
    cv::Rect roi_f_in_padded(0, 0, f.cols, f.rows); // THIS LINE IS MOVED/REDECLARED HERE

    cv::Mat f_cropped = f_updated_padded_real(kernel_roi).clone();
    f_cropped.convertTo(f, CV_32F);

    cv::threshold(f, f, 0.0f, 0.0f, cv::THRESH_TOZERO);

    cv::Scalar sum_val = cv::sum(f);
    if (sum_val[0] > std::numeric_limits<float>::epsilon()) {
        f = f / sum_val[0];
    } else {
        f = createInitialImpulseKernel(f.rows);
        std::cerr << "Warning: Kernel sum was zero/too small after F-step, re-initializing to impulse.\n";
    }

    current_padded_f = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Mat f_shifted = cv::Mat::zeros(f.size(), CV_32F);
    
    int cx = f.cols / 2;
    int cy = f.rows / 2;

    cv::Mat q0(f, cv::Rect(0, 0, cx, cy));         // Top-Left
    cv::Mat q1(f, cv::Rect(cx, 0, f.cols - cx, cy));    // Top-Right
    cv::Mat q2(f, cv::Rect(0, cy, cx, f.rows - cy));    // Bottom-Left
    cv::Mat q3(f, cv::Rect(cx, cy, f.cols - cx, f.rows - cy)); // Bottom-Right

    q0.copyTo(f_shifted(cv::Rect(f.cols - cx, f.rows - cy, cx, cy))); // TL to BR (of new shifted mat)
    q1.copyTo(f_shifted(cv::Rect(0, f.rows - cy, f.cols - cx, cy)));  // TR to BL
    q2.copyTo(f_shifted(cv::Rect(f.cols - cx, 0, cx, f.rows - cy)));  // BL to TR
    q3.copyTo(f_shifted(cv::Rect(0, 0, f.cols - cx, f.rows - cy)));   // BR to TL (this is where the impulse goes)

    f_shifted.copyTo(current_padded_f(roi_f_in_padded)); // This line now has roi_f_in_padded in scope
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
    cv::Mat result = cv::Mat::zeros(z.size(), z.type());

    cv::Mat abs_z;
    abs_z = cv::abs(z);

    cv::Mat abs_z_minus_tau = abs_z - tau;

    cv::Mat thresholded_abs_z;
    cv::max(abs_z_minus_tau, 0.0f, thresholded_abs_z);

    cv::Mat sign_z = cv::Mat::zeros(z.size(), z.type());

    z.forEach<float>([&](float &val, const int* position) {
        if (val > 0) {
            sign_z.at<float>(position[0], position[1]) = 1.0f;
        } else if (val < 0) {
            sign_z.at<float>(position[0], position[1]) = -1.0f;
        } else {
            sign_z.at<float>(position[0], position[1]) = 0.0f;
        }
    });

    result = sign_z.mul(thresholded_abs_z);
    return result;
}

void BlindDeblurrer::updateRho(const cv::Mat& L_x_current, const cv::Mat& L_y_current, const std::vector<cv::Mat>& f_Dk_current) {
    float rho_factor = 1.1f;
    float mu_balance = 10.0f;

    double r_norm_sq = 0.0;

    cv::Mat diff_Lx_hx = L_x_current - h_x;
    r_norm_sq += cv::norm(diff_Lx_hx, cv::NORM_L2SQR);

    cv::Mat diff_Ly_hy = L_y_current - h_y;
    r_norm_sq += cv::norm(diff_Ly_hy, cv::NORM_L2SQR);

    for (int k = 0; k < 6; ++k) {
        cv::Mat diff_fDk_pk = f_Dk_current[k] - p_k[k];
        r_norm_sq += cv::norm(diff_fDk_pk, cv::NORM_L2SQR);
    }
    double r_norm = std::sqrt(r_norm_sq);

    double s_norm_sq = 0.0;

    cv::Mat diff_hx_change = h_x - h_x_prev;
    s_norm_sq += cv::norm(diff_hx_change, cv::NORM_L2SQR);

    cv::Mat diff_hy_change = h_y - h_y_prev;
    s_norm_sq += cv::norm(diff_hy_change, cv::NORM_L2SQR);

    for (int k = 0; k < 6; ++k) {
        cv::Mat diff_pk_change = p_k[k] - p_k_prev[k];
        s_norm_sq += cv::norm(diff_pk_change, cv::NORM_L2SQR);
    }
    double s_norm = rho * std::sqrt(s_norm_sq);

    if (r_norm > mu_balance * s_norm) {
        rho *= rho_factor;
        std::cout << "Debug: Increasing rho. New rho: " << rho << std::endl;
    } else if (s_norm > mu_balance * r_norm) {
        rho /= rho_factor;
        std::cout << "Debug: Decreasing rho. New rho: " << rho << std::endl;
    } else {
        std::cout << "Debug: Rho unchanged. Rho: " << rho << std::endl;
    }

    float rho_min_val = 1e-3f;
    rho = std::max(rho_min_val, std::min(rho_max, rho));
    std::cout << "Current new Rho: " + std::to_string(rho) + "\n";
}

void BlindDeblurrer::updateAuxAndLagrange() {
    std::cout << "Debug: Entering updateAuxAndLagrange()...\n";

    h_x.copyTo(h_x_prev);
    h_y.copyTo(h_y_prev);
    for (int k = 0; k < 6; ++k) {
        p_k[k].copyTo(p_k_prev[k]);
    }

    cv::Mat L_x, L_y;
    computeGradientX(L, L_x);
    computeGradientY(L, L_y);

    cv::Mat term_h_x = L_x + lam_x / rho;
    h_x = shrink(term_h_x, lambda1 / rho);


    cv::Mat term_h_y = L_y + lam_y / rho;
    h_y = shrink(term_h_y, lambda1 / rho);

    lam_x = lam_x + rho * (L_x - h_x);
    lam_y = lam_y + rho * (L_y - h_y);

    std::vector<cv::Mat> f_Dk_rho(6);
    for (int k = 0; k < 6; ++k) {
        cv::Mat f_Dk;
        cv::filter2D(f, f_Dk, -1, spatial_deriv_kernels[k], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        double min_fDk, max_fDk;
        cv::minMaxLoc(f_Dk, &min_fDk, &max_fDk);

        f_Dk.copyTo(f_Dk_rho[k]);

        cv::Mat term_p_k = f_Dk + psi_k[k] / rho;
        p_k[k] = shrink(term_p_k, lambda2 / rho / zeta_k[k]);

        psi_k[k] = psi_k[k] + rho * (f_Dk - p_k[k]);
    }

    updateRho(L_x, L_y, f_Dk_rho);
    std::cout << "Debug: Exiting updateAuxAndLagrange() successfully.\n";
}