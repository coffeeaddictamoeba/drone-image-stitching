#include "../include/deblur.h"
#include "../include/dblrutils.h"
#include <limits>
#include <complex>


void BlindDeblurrer::setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size) {
    current_img_size = blurred_image_scale.size();

    if (initial_kernel_size % 2 == 0) initial_kernel_size++;

    L = blurred_image_scale.clone();
    f = createInitialImpulseKernel(initial_kernel_size);

    h_x = cv::Mat::zeros(current_img_size, CV_32F);
    h_y = cv::Mat::zeros(current_img_size, CV_32F);
    lam_x = cv::Mat::zeros(current_img_size, CV_32F);
    lam_y = cv::Mat::zeros(current_img_size, CV_32F);

    p_k.resize(6);
    psi_k.resize(6);
    for (int i = 0; i < 6; ++i) {
        p_k[i] = cv::Mat::zeros(current_img_size, CV_32F);
        psi_k[i] = cv::Mat::zeros(current_img_size, CV_32F);
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
    cv::Rect roi(0, 0, initial_kernel_size, initial_kernel_size);
    f.copyTo(current_padded_f(roi));

    computeDerivativeFiltersFFTs();
    computeLocalSmoothnessMask(blurred_image_scale);
}

LStepFFTInputsContainer BlindDeblurrer::prepareLStepInputsAndFFTs() {
    LStepFFTInputsContainer inputs;

    cv::Mat h_x_prime_unpadded = h_x - lam_x / rho;
    cv::Mat h_y_prime_unpadded = h_y - lam_y / rho;

    cv::Mat h_x_prime_padded = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Rect roi_h_x(0, 0, current_img_size.width, current_img_size.height);
    h_x_prime_unpadded.copyTo(h_x_prime_padded(roi_h_x));

    cv::Mat h_y_prime_padded = cv::Mat::zeros(current_dft_size, CV_32F);
    cv::Rect roi_h_y(0, 0, current_img_size.width, current_img_size.height);
    h_y_prime_unpadded.copyTo(h_y_prime_padded(roi_h_y));

    std::vector<cv::Mat> p_k_prime_padded(6);
    inputs.F_p_k_prime.resize(6);
    for (int k = 0; k < 6; ++k) {
        cv::Mat p_k_prime_unpadded = p_k[k] - psi_k[k] / rho;

        p_k_prime_padded[k] = cv::Mat::zeros(current_dft_size, CV_32F);
        cv::Rect roi_p_k(0, 0, current_img_size.width, current_img_size.height);
        p_k_prime_unpadded.copyTo(p_k_prime_padded[k](roi_p_k));
    }

    fft2d(h_x_prime_padded, inputs.F_h_x_prime);
    fft2d(h_y_prime_padded, inputs.F_h_y_prime);

    for (int k = 0; k < 6; ++k) {
        fft2d(p_k_prime_padded[k], inputs.F_p_k_prime[k]);
    }

    cv::Mat padded_blurred_32F, padded_f_32F;
    current_padded_blurred.convertTo(padded_blurred_32F, CV_32F);
    current_padded_f.convertTo(padded_f_32F, CV_32F);         
    
    fft2d(padded_blurred_32F, inputs.F_I);

    fft2d(padded_f_32F, inputs.F_f);

    cv::Mat F_f_planes_temp[2];
    cv::split(inputs.F_f, F_f_planes_temp);
    F_f_planes_temp[1] = -F_f_planes_temp[1];
    cv::Mat F_f_conj_32FC2;
    cv::merge(F_f_planes_temp, 2, F_f_conj_32FC2);
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
    std::cout << "Debug: denominator_fft_real_before_max size: " << denominator_fft_real_before_max.size() << ", depth: " << denominator_fft_real_before_max.depth() << std::endl;

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

void BlindDeblurrer::padImageForDFT(const cv::Mat& input, cv::Mat& padded_output, cv::Size& dft_size) {
    dft_size = getOptimalDFTSize(input.size()); // Use the utility function
    cv::copyMakeBorder(input, padded_output, 0, dft_size.height - input.rows, 0, dft_size.width - input.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
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

FStepFFTInputsContainer BlindDeblurrer::prepareFStepInputsAndFFTs() {
    FStepFFTInputsContainer inputs;

    cv::Mat current_padded_L_64F;
    current_padded_L.convertTo(current_padded_L_64F, CV_64F);
    debugMatrix(current_padded_L, "current_padded_L");
    std::cout << "Debug: current_padded_L size: " << current_padded_L.size() << ", depth: " << current_padded_L.depth() << std::endl; // Add this
    std::cout << "Debug: current_padded_L_64F size: " << current_padded_L_64F.size() << ", depth: " << current_padded_L_64F.depth() << std::endl; // Add this
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
        cv::Rect roi_p_k(0, 0, current_img_size.width, current_img_size.height);
        p_k[k].convertTo(p_k_padded_64F(roi_p_k), CV_64F);

        fft2d(p_k_padded_64F, inputs.F_p_k[k]);
    }
    return inputs;
}

cv::Mat BlindDeblurrer::computeFNumerator(const FStepFFTInputsContainer& inputs) {
    // Term 1: F_L_conj * F_I
    cv::Mat term1_num;
    // mulSpectrums(src1, src2, dst, flags, conj_src2)
    // We want F_L_conj * F_I. So, src1=F_I, src2=F_L, conj_src2=true (or src1=F_L_conj, src2=F_I, conj_src2=false)
    cv::mulSpectrums(inputs.F_I_64FC2, inputs.F_L, term1_num, 0, true); // (A * conj(B)) -> (F_I * conj(F_L))

    cv::Mat sum_of_Dk_Pk_terms = cv::Mat::zeros(current_dft_size, CV_64FC2);

    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);

        cv::Mat F_deriv_k_conj;
        cv::Mat F_deriv_k_planes_temp[2];
        cv::split(F_deriv_k_64FC2, F_deriv_k_planes_temp);
        F_deriv_k_planes_temp[1] = -F_deriv_k_planes_temp[1];
        cv::merge(F_deriv_k_planes_temp, 2, F_deriv_k_conj);

        cv::Mat temp_term; // F_Dk_conj * F_Pk
        // conj(F_Dk) * F_Pk. So src1=F_Pk, src2=F_Dk, conj_src2=true
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
    cv::magnitude(F_L_real_part, F_L_imag_part, F_L_mag_sq);
    F_L_mag_sq = F_L_mag_sq.mul(F_L_mag_sq);

    // Term 2: rho * SUM(|F_Dk|^2)
    cv::Mat sum_of_Dk_mag_sq = cv::Mat::zeros(current_dft_size, CV_64F);

    for (int k = 0; k < 6; ++k) {
        cv::Mat F_deriv_k_64FC2;
        this->F_deriv_filters[k].convertTo(F_deriv_k_64FC2, CV_64FC2);

        std::cout << "Debug: F_deriv_filters[" << k << "] (original) size: " << this->F_deriv_filters[k].size() << ", depth: " << this->F_deriv_filters[k].depth() << std::endl;
        std::cout << "Debug: F_deriv_filters[" << k << "] (converted to CV_64FC2) size: " << F_deriv_k_64FC2.size() << ", depth: " << F_deriv_k_64FC2.depth() << std::endl;

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

    cv::Rect kernel_roi(0, 0, f.cols, f.rows);

    f = f_updated_padded_real(kernel_roi).clone();
    f.convertTo(f, CV_32F);

    cv::threshold(f, f, 0.0f, 0.0f, cv::THRESH_TOZERO);

    cv::Scalar sum_val = cv::sum(f);
    if (sum_val[0] > std::numeric_limits<float>::epsilon()) {
        f = f / sum_val[0];
    } else {
        f = createInitialImpulseKernel(f.rows);
        std::cerr << "Warning: Kernel sum was zero/too small after F-step, re-initializing to impulse.\n";
    }
}

void BlindDeblurrer::updateFStep() {
    std::cout << "Executing one f-step iteration...\n";

    FStepFFTInputsContainer inputs = prepareFStepInputsAndFFTs();

    cv::Mat numerator_f_fft = computeFNumerator(inputs);
    cv::Mat denominator_f_fft_complex = computeFDenominator(inputs);
    
    double min_f_denom_mag, max_f_denom_mag;
    cv::minMaxLoc(cv::abs(denominator_f_fft_complex), &min_f_denom_mag, &max_f_denom_mag);
    std::cout << "Debug: F-step Denominator FFT (complex) - Min Mag: " << min_f_denom_mag << ", Max Mag: " << max_f_denom_mag << std::endl;

    cv::Mat F_f_updated;
    F_f_updated = divideComplex(numerator_f_fft, denominator_f_fft_complex);

    std::cout << "Debug: F_f_updated (After division) nans & infs:\n"; checkInfNan(F_f_updated, "F_f_updated");
    double min_f_updated, max_f_updated;
    cv::minMaxLoc(F_f_updated, &min_f_updated, &max_f_updated); // Checks real part magnitude for complex mat
    std::cout << "Debug: F_f_updated (After division) - Min Val: " << min_f_updated << ", Max Val: " << max_f_updated << std::endl;

    performFStepIFFTAndPostProcess(F_f_updated);

    std::cout << "Debug: updateFStep() completed successfully." << std::endl;
}