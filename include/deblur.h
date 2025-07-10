#include <opencv4/opencv2/opencv.hpp>
#include <vector>

struct LStepFFTInputsContainer {
    cv::Mat F_h_x_prime;
    cv::Mat F_h_y_prime;
    std::vector<cv::Mat> F_p_k_prime;
    cv::Mat F_I; // FFT of padded blurred image
    cv::Mat F_f; // FFT of padded kernel
    cv::Mat F_f_conj_64FC2; // Conjugate of F_f, converted to CV_64FC2 for multiplication
};

struct FStepFFTInputsContainer {
    cv::Mat F_L;         // FFT of padded L (latent image)
    cv::Mat F_L_conj;    // Conjugate of F_L
    cv::Mat F_I_64FC2;   // FFT of padded blurred image (ensured CV_64FC2)
    std::vector<cv::Mat> F_p_k; // FFTs of padded p_k auxiliary variables
};

class BlindDeblurrer {
public:
    cv::Mat L; // Latent image estimate
    cv::Mat f; // Blur kernel estimate

    // ADMM auxiliary variables and Lagrange multipliers
    cv::Mat h_x, h_y; // Auxiliary variables for L's gradients
    cv::Mat lam_x, lam_y; // Lagrange multipliers for L's gradients
    std::vector<cv::Mat> p_k; // Auxiliary variables for noise derivatives
    std::vector<cv::Mat> psi_k; // Lagrange multipliers for noise derivatives

    // Previous auxiliary variables for rho update ---
    cv::Mat h_x_prev, h_y_prev;
    std::vector<cv::Mat> p_k_prev;

    // Noise parameters
    std::vector<float> zeta_k;

    // Optimization parameters
    float rho;
    float lambda1;
    float lambda2;
    float sigma1;
    float rho_max;

    // Current image and DFT dimensions
    cv::Size current_img_size;
    cv::Size current_dft_size;

    // Padded versions for FFT operations
    cv::Mat current_padded_blurred;
    cv::Mat current_padded_L;
    cv::Mat current_padded_f; // Padded kernel f

    // Pre-computed FFTs of derivative filters
    std::vector<cv::Mat> F_deriv_filters;

    // Pre-computed spatial derivative kernels (for h_x, h_y, p_k updates)
    std::vector<cv::Mat> spatial_deriv_kernels;

    // Local smoothness mask
    cv::Mat M_mask;

    // Constructor
    BlindDeblurrer() {
        lambda1 = 0.005f;
        lambda2 = 0.05f;
        rho = 0.1f;
        sigma1 = 0.02f;
        rho_max = 100.0f;
    };

    void setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size);

    void updateLStep();
    void updateFStep();
    void updateAuxAndLagrange();

private:
    cv::Mat createInitialImpulseKernel(int size);
    void computeGradientX(const cv::Mat& input, cv::Mat& output);
    void computeGradientY(const cv::Mat& input, cv::Mat& output);
    void computeDerivativeFiltersFFTs();

    void computeLocalSmoothnessMask(const cv::Mat& blurred_img_scale);

    void initializeSpatialDerivativeKernels();

    // L-step helpers
    cv::Mat padToSize(const cv::Mat& input, const cv::Size& target_size);
    void performLStepIFFTAndPostProcess(const cv::Mat& F_L_updated);
    cv::Mat computeLDenominator(const cv::Mat& F_f);
    cv::Mat computeLNumerator(const LStepFFTInputsContainer& inputs);
    LStepFFTInputsContainer prepareLStepInputsAndFFTs();

    // F-step helpers
    FStepFFTInputsContainer prepareFStepInputsAndFFTs();
    cv::Mat computeFNumerator(const FStepFFTInputsContainer& inputs);
    cv::Mat computeFDenominator(const FStepFFTInputsContainer& inputs);
    void performFStepIFFTAndPostProcess(const cv::Mat& F_f_updated);

    cv::Mat shrink(const cv::Mat& z, float tau);
    void updateRho(const cv::Mat& L_x_current, const cv::Mat& L_y_current, const std::vector<cv::Mat>& f_Dk_current);
};