#include "../include/deblur.h"
#include "../include/dblrutils.h"
#include <limits>
#include <complex>
#include <string>


void BlindDeblurrer::setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size) {
    current_img_size = blurred_image_scale.size();

    if (initial_kernel_size % 2 == 0) ++initial_kernel_size; // Ensure kernel size is odd

    // Convert input image to [0,1] float
    cv::Mat blurred_normalized;
    switch (blurred_image_scale.depth()) {
        case CV_8U:
            blurred_image_scale.convertTo(blurred_normalized, CV_32F, 1.0 / 255.0);
            break;
        case CV_16U:
            blurred_image_scale.convertTo(blurred_normalized, CV_32F, 1.0 / 65535.0);
            break;
        default:
            blurred_image_scale.convertTo(blurred_normalized, CV_32F);
    }

    // Initialize L
    cv::Mat L_init;
    cv::Laplacian(blurred_normalized, L_init, CV_32F, 3);
    L = blurred_normalized - 0.5f * L_init;
    cv::threshold(L, L, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(L, L, 1.0f, 1.0f, cv::THRESH_TRUNC);

    // Initialize kernel f
    f = createInitialImpulseKernel(initial_kernel_size);
    cv::Size kernel_size = f.size();

    // Initialize auxiliary variables and Lagrange multipliers
    auto zeroImage = [&](cv::Size sz) { return cv::Mat::zeros(sz, CV_32F); };
    h_x = zeroImage(current_img_size);
    h_y = zeroImage(current_img_size);
    lam_x = zeroImage(current_img_size);
    lam_y = zeroImage(current_img_size);
    h_x_prev = zeroImage(current_img_size);
    h_y_prev = zeroImage(current_img_size);

    p_k.resize(6); psi_k.resize(6); p_k_prev.resize(6); zeta_k.assign(6, 1.0f);
    for (int i = 0; i < 6; ++i) {
        p_k[i] = zeroImage(kernel_size);
        psi_k[i] = zeroImage(kernel_size);
        p_k_prev[i] = zeroImage(kernel_size);
    }

    // Compute optimal DFT size and pad images
    current_dft_size = getOptimalDFTSize(
        cv::Size(current_img_size.width + initial_kernel_size - 1,
                 current_img_size.height + initial_kernel_size - 1)
    );

    current_padded_blurred = padToSize(blurred_normalized, current_dft_size);
    current_padded_L = padToSize(L, current_dft_size);

    // Initialize and place kernel into padded f
    current_padded_f = cv::Mat::zeros(current_dft_size, CV_32F);
    f.copyTo(current_padded_f(cv::Rect(0, 0, kernel_size.width, kernel_size.height)));

    initializeSpatialDerivativeKernels();
    computeDerivativeFiltersFFTs();
    computeLocalSmoothnessMask(blurred_normalized);  // use normalized input
}

// Create a 2D Gaussian kernel centered at (center, center)
cv::Mat BlindDeblurrer::createInitialImpulseKernel(int size) {
    CV_Assert(size % 2 == 1);
    cv::Mat kernel(size, size, CV_32F);

    const int center = size / 2;
    const float sigma2 = (size / 6.0f) * (size / 6.0f);

    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            float dx = x - center;
            float dy = y - center;
            kernel.at<float>(y, x) = std::exp(-(dx * dx + dy * dy) / (2 * sigma2));
        }

    kernel /= cv::sum(kernel)[0];
    return kernel;
}

void centerKernel(cv::Mat& kernel) {
    cv::Mat centered;
    int cx = kernel.cols / 2;
    int cy = kernel.rows / 2;

    // Shift quadrants
    cv::Mat q0(kernel, cv::Rect(0, 0, cx, cy)); // Top-left
    cv::Mat q1(kernel, cv::Rect(cx, 0, cx, cy)); // Top-right
    cv::Mat q2(kernel, cv::Rect(0, cy, cx, cy)); // Bottom-left
    cv::Mat q3(kernel, cv::Rect(cx, cy, cx, cy)); // Bottom-right

    // Swap diagonals
    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

void BlindDeblurrer::computeGradientX(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel_x = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
    cv::filter2D(input, output, -1, kernel_x, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

void BlindDeblurrer::computeGradientY(const cv::Mat& input, cv::Mat& output) {
    cv::Mat kernel_y = (cv::Mat_<float>(3, 1) << -0.5, 0, 0.5);
    cv::filter2D(input, output, -1, kernel_y, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
}

// Approximate gradient of global image prior w.r.t L
cv::Mat BlindDeblurrer::computeGlobalPriorGradient(const cv::Mat& L_in) {
    std::cout << "[Global Prior] Start computing global prior.\n";
    cv::Mat grad_x, grad_y;
    computeGradientX(L_in, grad_x);
    computeGradientY(L_in, grad_y);

    cv::Mat dphi_x = grad_x.clone();
    cv::Mat dphi_y = grad_y.clone();

    for (int y = 0; y < grad_x.rows; ++y) {
        for (int x = 0; x < grad_x.cols; ++x) {
            dphi_x.at<float>(y, x) = dphi(grad_x.at<float>(y, x));
            dphi_y.at<float>(y, x) = dphi(grad_y.at<float>(y, x));
        }
    }

    // Backproject to image space using -div operator (approximate adjoint of gradient)
    cv::Mat div_x, div_y;
    computeGradientX(dphi_x, div_x); // forward diff of adjoint (Dx^T)
    computeGradientY(dphi_y, div_y); // forward diff of adjoint (Dy^T)

    cv::Mat grad = -(div_x + div_y);
    grad.convertTo(grad, CV_32F);  // Ensure correct type
    return grad;
}

cv::Mat BlindDeblurrer::padToSize(const cv::Mat& input, const cv::Size& target_size) {
    cv::Mat padded;
    int pad_bottom = target_size.height - input.rows;
    int pad_right = target_size.width - input.cols;

    cv::copyMakeBorder(input, padded, 0, pad_bottom, 0, pad_right, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    return padded;
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
    cv::Rect roi_img(0, 0, current_img_size.width, current_img_size.height);
    cv::Rect roi_ker(0, 0, p_k[0].cols, p_k[0].rows);

    cv::Mat h_x_prime = h_x - lam_x / rho;
    cv::Mat h_y_prime = h_y - lam_y / rho;

    // Pad for FFT
    inputs.F_h_x_prime = fft2dWithPad(h_x_prime, current_dft_size, roi_img);
    inputs.F_h_y_prime = fft2dWithPad(h_y_prime, current_dft_size, roi_img);

    inputs.F_p_k_prime.resize(6);
    for (int k = 0; k < 6; ++k) {
        cv::Mat pk_prime = p_k[k] - psi_k[k] / rho;
        inputs.F_p_k_prime[k] = fft2dWithPad(pk_prime, current_dft_size, roi_ker);
    }

    // Image and kernel FFTs
    fft2d(current_padded_blurred, inputs.F_I);
    fft2d(current_padded_f, inputs.F_f);

    // Precompute F_f conjugate
    inputs.F_f_conj_64FC2 = conj64F(inputs.F_f);
    return inputs;
}

cv::Mat BlindDeblurrer::computeLNumerator(const LStepFFTInputsContainer& in) {
    std::cout << "[L-Step] Start computing L numerator.\n";

    // Term 1: conj(F_f) * F_I
    cv::Mat F_I_64F;
    in.F_I.convertTo(F_I_64F, CV_64FC2);
    cv::Mat term1;
    cv::mulSpectrums(F_I_64F, in.F_f_conj_64FC2, term1, 0, false);

    // Term 2: conj(F_Dx) * F_h_x_prime + conj(F_Dy) * F_h_y_prime
    cv::Mat F_Dx_conj = conj64F(F_deriv_filters[1]);
    cv::Mat F_Dy_conj = conj64F(F_deriv_filters[2]);

    cv::Mat F_h_x_64F, F_h_y_64F;
    in.F_h_x_prime.convertTo(F_h_x_64F, CV_64FC2);
    in.F_h_y_prime.convertTo(F_h_y_64F, CV_64FC2);

    cv::Mat term2x, term2y;
    cv::mulSpectrums(F_h_x_64F, F_Dx_conj, term2x, 0, false);
    cv::mulSpectrums(F_h_y_64F, F_Dy_conj, term2y, 0, false);
    cv::Mat term2 = rho * (term2x + term2y);

    // Term 3: sum_k conj(F_Dk) * F_p_k_prime
    cv::Mat term3 = cv::Mat::zeros(current_dft_size, CV_64FC2);
    for (int k = 0; k < 6; ++k) {
        cv::Mat F_Dk_conj = conj64F(F_deriv_filters[k]);
        cv::Mat F_p_k_64F;
        in.F_p_k_prime[k].convertTo(F_p_k_64F, CV_64FC2);

        cv::Mat prod;
        cv::mulSpectrums(F_p_k_64F, F_Dk_conj, prod, 0, false);
        term3 += prod;
    }
    term3 *= rho;

    // Term 4: gradient of global prior (spatial domain, added after IFFT)
    global_prior_grad = computeGlobalPriorGradient(L);
    std::cout << "[L-Step] End computing L numerator.\n";
    return term1 + term2 + term3;
}

cv::Mat BlindDeblurrer::computeLDenominator(const cv::Mat& F_f) {
    std::cout << "[L-Step] Start computing L denominator.\n";
    cv::Mat F_f_64F;
    F_f.convertTo(F_f_64F, CV_64FC2);

    // Term 1: |F_f|^2
    cv::Mat f_mag = magSq64F(F_f_64F);

    // Term 2: rho * (|F_Dx|^2 + |F_Dy|^2)
    cv::Mat D_x = magSq64F(F_deriv_filters[1]);
    cv::Mat D_y = magSq64F(F_deriv_filters[2]);
    cv::Mat term2 = rho * (D_x + D_y);

    // Term 3: rho * sum_k |F_Dk|^2
    cv::Mat term3 = cv::Mat::zeros(current_dft_size, CV_64F);
    for (int k = 0; k < 6; ++k) {
        term3 += magSq64F(F_deriv_filters[k]);
    }
    term3 *= rho;

    cv::Mat denom_real = f_mag + term2 + term3;
    cv::max(denom_real, 1e-6, denom_real); // stability

    // Merge into complex with 0 imaginary
    std::cout << "[L-Step] End computing L denominator.\n";
    return mergeRealImag(denom_real, cv::Mat::zeros(current_dft_size, CV_64F));
}

void BlindDeblurrer::performLStepIFFTAndPostProcess(const cv::Mat& F_L_updated) {
    std::cout << "[L-Step] Start computing L IFFT.\n";
    cv::Mat spatial_complex;
    ifft2d(F_L_updated, spatial_complex);

    std::vector<cv::Mat> planes;
    cv::split(spatial_complex, planes);

    if (!planes[1].empty()) {
        double imag_max;
        cv::minMaxLoc(cv::abs(planes[1]), nullptr, &imag_max);
        CV_Assert(imag_max < 1e-3 && "IFFT imaginary part too large — instability?");
    }

    cv::Mat L_updated = planes[0](cv::Rect(0, 0, current_img_size.width, current_img_size.height)).clone();

    if (global_prior_grad.size() != L_updated.size() || global_prior_grad.type() != CV_32F) {
        std::cerr << "[ERROR] global_prior_grad has incompatible size or type.\n";
        std::cerr << "Expected: " << L_updated.size() << " CV_32F, Got: "
                  << global_prior_grad.size() << " " << global_prior_grad.type() << "\n";
        return;
    }    

    // Add spatial global prior gradient
    cv::Mat prior_weight = 1.0f - M_mask;
    cv::Mat masked_prior = global_prior_grad.mul(prior_weight);
    L_updated -= lambda1 * masked_prior;

    cv::threshold(L_updated, L_updated, 0.0f, 0.0f, cv::THRESH_TOZERO);
    cv::threshold(L_updated, L_updated, 1.0f, 1.0f, cv::THRESH_TRUNC);

    L = L_updated;
    current_padded_L = padToSize(L, current_dft_size);
    std::cout << "[L-Step] End computing L IFFT.\n";
}

void BlindDeblurrer::updateLStep() {
    std::cout << "[L-Step] Starting...\n";

    auto inputs = prepareLStepInputsAndFFTs();
    auto numerator_fft = computeLNumerator(inputs);
    auto denominator_fft = computeLDenominator(inputs.F_f);

    cv::Mat F_L_updated = divideComplex(numerator_fft, denominator_fft);

    if (!cv::checkRange(F_L_updated)) {
        std::cerr << "[ERROR] F_L_updated contains NaNs or infs!\n";
        return;
    }

    performLStepIFFTAndPostProcess(F_L_updated);
    std::cout << "[L-Step] Completed.\n";
}

// --- F-Step ---
FStepFFTInputsContainer BlindDeblurrer::prepareFStepInputsAndFFTs() {
    FStepFFTInputsContainer inputs;
    cv::Rect roi_kernel(0, 0, f.cols, f.rows);

    // Convert and FFT padded L
    cv::Rect roi_L(0, 0, current_padded_L.cols, current_padded_L.rows);
    inputs.F_L = fft2dWithPad(current_padded_L, current_dft_size, roi_L);

    // Convert and FFT padded blurred image
    cv::Mat blurred_64F;
    current_padded_blurred.convertTo(blurred_64F, CV_64F);
    fft2d(blurred_64F, inputs.F_I_64FC2);

    // FFT of padded p_k
    inputs.F_p_k.resize(6);
    for (int k = 0; k < 6; ++k) {
        inputs.F_p_k[k] = fft2dWithPad(p_k[k], current_dft_size, roi_kernel);
    }

    return inputs;
}

cv::Mat BlindDeblurrer::computeFNumerator(const FStepFFTInputsContainer& in) {
    // Term 1: conj(F_L) * F_I
    cv::Mat F_L_conj = conj64F(in.F_L);
    cv::Mat term1;
    cv::mulSpectrums(F_L_conj, in.F_I_64FC2, term1, 0, false);

    // Term 2: sum_k conj(F_Dk) * F_p_k
    cv::Mat term2 = cv::Mat::zeros(current_dft_size, CV_64FC2);
    for (int k = 0; k < 6; ++k) {
        cv::Mat F_Dk_conj = conj64F(F_deriv_filters[k]);
        cv::Mat F_p_k_64F;
        in.F_p_k[k].convertTo(F_p_k_64F, CV_64FC2);

        cv::Mat prod;
        cv::mulSpectrums(F_p_k_64F, F_Dk_conj, prod, 0, false);
        term2 += prod;
    }

    return term1 + rho * term2;
}

cv::Mat BlindDeblurrer::computeFDenominator(const FStepFFTInputsContainer& in) {
    // Term 1: |F_L|^2
    cv::Mat mag_L_sq = magSq64F(in.F_L);

    // Term 2: rho * sum_k |F_Dk|^2
    cv::Mat sum_deriv_mag_sq = cv::Mat::zeros(current_dft_size, CV_64F);
    for (int k = 0; k < 6; ++k) {
        sum_deriv_mag_sq += magSq64F(F_deriv_filters[k]);
    }

    cv::Mat denom_real = mag_L_sq + rho * sum_deriv_mag_sq;
    cv::max(denom_real, 1e-6, denom_real);

    return mergeRealImag(denom_real, cv::Mat::zeros(current_dft_size, CV_64F));
}

void BlindDeblurrer::performFStepIFFTAndPostProcess(const cv::Mat& F_f_updated) {
    cv::Mat spatial_complex;
    ifft2d(F_f_updated, spatial_complex);

    std::vector<cv::Mat> planes(2);
    cv::split(spatial_complex, planes);
    CV_Assert(planes[0].depth() == CV_64F);

    // Use only real part
    cv::Mat f_real = planes[0](cv::Rect(0, 0, f.cols, f.rows)).clone();
    f_real.convertTo(f, CV_32F);

    cv::threshold(f, f, 0.0f, 0.0f, cv::THRESH_TOZERO); // only positive

    double sum_f = cv::sum(f)[0];
    if (sum_f <= std::numeric_limits<float>::epsilon()) {
        std::cerr << "[F-Step] Kernel degenerated. Reinitializing.\n";
        f = createInitialImpulseKernel(f.rows);
    } else {
        f /= sum_f;
    }

    // Optional: regularization to encourage center bias
    f = f.mul(gaussianWindow(f.rows));
    f /= cv::sum(f)[0];

    current_padded_f = cv::Mat::zeros(current_dft_size, CV_32F);
    f.copyTo(current_padded_f(cv::Rect(0, 0, f.cols, f.rows)));
}

void BlindDeblurrer::updateFStep() {
    std::cout << "[F-Step] Starting update...\n";

    FStepFFTInputsContainer inputs = prepareFStepInputsAndFFTs();

    cv::Mat numerator_fft = computeFNumerator(inputs);
    cv::Mat denominator_fft = computeFDenominator(inputs);

    cv::Mat F_f_updated = divideComplex(numerator_fft, denominator_fft);
    if (!cv::checkRange(F_f_updated)) {
        std::cerr << "[F-Step] Error: Invalid FFT values (NaN or Inf).\n";
        return;
    }

    performFStepIFFTAndPostProcess(F_f_updated);

    double min_f, max_f;
    cv::minMaxLoc(f, &min_f, &max_f);
    std::cout << "[F-Step] Kernel updated. Min: " << min_f << ", Max: " << max_f
              << ", Sum: " << cv::sum(f)[0] << ", Norm: " << cv::norm(f) << "\n";
}

// --- Parameters update ---
cv::Mat BlindDeblurrer::shrink(const cv::Mat& z, float tau) {
    CV_Assert(z.type() == CV_32F);

    cv::Mat abs_z, sign_z, thresholded;
    cv::absdiff(z, cv::Scalar(0), abs_z); // |z|
    sign_z = cv::Mat::zeros(z.size(), z.type());
    sign_z.setTo(1.0f, z > 0);
    sign_z.setTo(-1.0f, z < 0);

    // (|z| - tau)_+
    cv::subtract(abs_z, tau, thresholded);
    cv::max(thresholded, 0, thresholded);

    return sign_z.mul(thresholded);
}

void BlindDeblurrer::updateRho(const cv::Mat&, const cv::Mat&, const std::vector<cv::Mat>&) {
    // Fixed for now, adaptive strategy can be added later
    rho = 0.1f;
    std::cout << "[Rho Update] Using fixed rho = " << rho << "\n";
}

void BlindDeblurrer::updateAuxAndLagrange() {
    std::cout << "[ADMM] Updating auxiliary and Lagrange variables...\n";

    h_x.copyTo(h_x_prev);
    h_y.copyTo(h_y_prev);
    for (int k = 0; k < 6; ++k)
        p_k[k].copyTo(p_k_prev[k]);

    // Compute gradients L
    cv::Mat L_x, L_y;
    computeGradientX(L, L_x);
    computeGradientY(L, L_y);

    const float tau1 = lambda1 / rho;

    // Scale lambda by rho
    cv::Mat lam_x_scaled, lam_y_scaled;
    cv::divide(lam_x, rho, lam_x_scaled, 1, CV_32F);
    cv::divide(lam_y, rho, lam_y_scaled, 1, CV_32F);

    // Ensure gradient is CV_32F
    L_x.convertTo(L_x, CV_32F);
    L_y.convertTo(L_y, CV_32F);

    // Soft-threshold step for h_x, h_y
    h_x = shrink(L_x + lam_x_scaled, tau1);
    h_y = shrink(L_y + lam_y_scaled, tau1);

    // Dual variable updates lam_x, lam_y
    lam_x = lam_x + rho * (L_x - h_x);
    lam_y = lam_y + rho * (L_y - h_y);

    // Update f-derivatives and auxiliary vars p_k, psi_k
    std::vector<cv::Mat> f_Dk_current(6);
    for (int k = 0; k < 6; ++k) {
        cv::filter2D(f, f_Dk_current[k], -1, spatial_deriv_kernels[k], cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

        const float tau2 = std::max(1e-4f, lambda2 / (rho * zeta_k[k]));

        cv::Mat psi_scaled;
        cv::divide(psi_k[k], rho, psi_scaled, 1, CV_32F);

        cv::Mat fk_plus = f_Dk_current[k] + psi_scaled;
        p_k[k] = shrink(fk_plus, tau2);

        psi_k[k] = psi_k[k] + rho * (f_Dk_current[k] - p_k[k]);
    }

    double min_f, max_f;
    cv::minMaxLoc(f, &min_f, &max_f);
    std::cout << "[Kernel Stats] f min: " << min_f << ", max: " << max_f
              << ", sum: " << cv::sum(f)[0] << "\n";

    updateRho(L_x, L_y, f_Dk_current);
    std::cout << "[ADMM] Rho: " << rho << ", Lambda1/rho: " << tau1 << "\n";
    std::cout << "[ADMM] updateAuxAndLagrange() complete.\n";
}