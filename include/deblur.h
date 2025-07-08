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

    // Noise parameters
    std::vector<float> zeta_k;

    // Optimization parameters
    float rho;
    float tau;
    float lambda1;
    float lambda2;
    float sigma1;

    // Current image and DFT dimensions
    cv::Size current_img_size;
    cv::Size current_dft_size;

    // Padded versions for FFT operations
    cv::Mat current_padded_blurred;
    cv::Mat current_padded_L;
    cv::Mat current_padded_f; // Padded kernel f

    // Pre-computed FFTs of derivative filters
    std::vector<cv::Mat> F_deriv_filters;

    // Local smoothness mask
    cv::Mat M_mask;

    // Constructor
    BlindDeblurrer() {
        rho = 0.1f;
        tau = 0.001f;
        lambda1 = 1.0f;
        lambda2 = 1.0f;
        sigma1 = 0.02f;
    };

    /**
     * @brief Sets up all variables and parameters for a given image scale.
     * @param blurred_image_scale The blurred image at the current scale.
     * @param initial_kernel_size The initial guess for the blur kernel dimension.
     */
    void setupForScale(const cv::Mat& blurred_image_scale, int initial_kernel_size);

    void updateLStep();
    void updateFStep();

private:
    /**
     * @brief Creates an initial impulse kernel.
     * @param size The dimension of the square kernel.
     */
    static cv::Mat createInitialImpulseKernel(int size);

    /**
     * @brief Computes the gradient in X-direction.
     */
    void computeGradientX(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief Computes the gradient in Y-direction.
     */
    void computeGradientY(const cv::Mat& input, cv::Mat& output);

    /**
     * @brief Computes and stores the FFTs of standard derivative filters.
     */
    void computeDerivativeFiltersFFTs();

    /**
     * @brief Computes the local smoothness mask (M_mask).
     */
    void computeLocalSmoothnessMask(const cv::Mat& blurred_img_scale);

    // L-step helpers
    void padImageForDFT(const cv::Mat& input, cv::Mat& padded_output, cv::Size& dft_size);
    void performLStepIFFTAndPostProcess(const cv::Mat& F_L_updated);
    cv::Mat computeLDenominator(const cv::Mat& F_f);
    cv::Mat computeLNumerator(const LStepFFTInputsContainer& inputs);
    LStepFFTInputsContainer prepareLStepInputsAndFFTs();

    // F-step helpers
    FStepFFTInputsContainer prepareFStepInputsAndFFTs();
    cv::Mat computeFNumerator(const FStepFFTInputsContainer& inputs);
    cv::Mat computeFDenominator(const FStepFFTInputsContainer& inputs);
    void performFStepIFFTAndPostProcess(const cv::Mat& F_f_updated);
};