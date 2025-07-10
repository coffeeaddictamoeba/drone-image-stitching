#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <complex>
#include <limits>

/**
 * @file dblrutils.h
 * @brief Utility functions for blind deconvolution and related image processing tasks.
 *
 * This header provides helper functions for FFT/IFFT operations, complex number arithmetic
 * on OpenCV matrices, mathematical calculations, and debugging utilities.
 */

// --- Calculation helpers ---

/**
 * @brief Calculates the optimal DFT size for a given input image size.
 *
 * OpenCV's DFT performance is optimized for arrays whose sizes are powers of 2, 3, and 5.
 * This function finds the smallest size that is greater than or equal to the input size
 * and is suitable for efficient DFT computation.
 *
 * @param size The original size (width, height) of the image.
 * @return cv::Size The optimal DFT size (width, height).
 */
cv::Size getOptimalDFTSize(const cv::Size& size);

/**
 * @brief Performs a 2D Fast Fourier Transform (FFT) on a single-channel real matrix.
 *
 * The input real matrix is converted to a 2-channel complex matrix (with zero imaginary parts)
 * before applying the DFT. The output is a 2-channel complex matrix representing the
 * frequency domain.
 *
 * @param input_real The single-channel input matrix (e.g., CV_32F, CV_64F).
 * @param output_complex The 2-channel complex matrix where the FFT result will be stored (e.g., CV_64FC2).
 * @pre input_real must have 1 channel.
 */
void fft2d(const cv::Mat& input_real, cv::Mat& output_complex);

/**
 * @brief Performs a 2D Inverse Fast Fourier Transform (IFFT) on a 2-channel complex matrix.
 *
 * The IFFT operation includes scaling to bring the frequency domain data back to the
 * spatial domain's original magnitude. The output is a 2-channel complex matrix.
 * To get the real spatial image, the real part of the output should be extracted.
 *
 * @param input_complex The 2-channel complex input matrix (e.g., CV_64FC2).
 * @param output_complex The 2-channel complex matrix where the IFFT result will be stored (e.g., CV_64FC2).
 * @pre input_complex must have 2 channels.
 */
void ifft2d(const cv::Mat& input_complex, cv::Mat& output_complex);

/**
 * @brief Pads a source image to a specified size with zeros and then computes its 2D FFT.
 *
 * This function creates a zero-padded matrix of the `size` dimensions, copies the
 * `src` image into the specified `roi` (Region of Interest) of the padded matrix,
 * and then performs a 2D FFT on the result.
 *
 * @param src The single-channel source image (e.g., CV_32F).
 * @param size The desired size of the padded matrix before FFT.
 * @param roi The region within the padded matrix where the source image will be copied.
 * @return cv::Mat A 2-channel complex matrix containing the FFT of the padded image.
 */
cv::Mat fft2dWithPad(const cv::Mat& src, cv::Size size, const cv::Rect& roi);

/**
 * @brief Computes the complex conjugate of a 2-channel complex matrix.
 *
 * Converts the input matrix to CV_64FC2 if it's not already, then negates
 * the imaginary part of each complex number.
 *
 * @param complex32 The input 2-channel complex matrix (e.g., CV_32FC2, CV_64FC2).
 * @return cv::Mat A new 2-channel CV_64FC2 matrix containing the complex conjugate.
 */
cv::Mat conj64F(const cv::Mat& complex32);

/**
 * @brief Computes the squared magnitude (real part^2 + imaginary part^2) of a 2-channel complex matrix.
 *
 * Converts the input matrix to CV_64FC2 if it's not already, then calculates
 * the squared magnitude for each element.
 *
 * @param complex32 The input 2-channel complex matrix (e.g., CV_32FC2, CV_64FC2).
 * @return cv::Mat A new single-channel CV_64F matrix containing the squared magnitudes.
 */
cv::Mat magSq64F(const cv::Mat& complex32);

/**
 * @brief Merges a real matrix and an imaginary matrix into a single 2-channel complex matrix.
 *
 * @param real The single-channel real part matrix.
 * @param imag The single-channel imaginary part matrix.
 * @return cv::Mat A new 2-channel complex matrix.
 * @pre real and imag must have the same size and type.
 */
cv::Mat mergeRealImag(const cv::Mat& real, const cv::Mat& imag);

/**
 * @brief Performs element-wise complex division of two 2-channel complex matrices.
 *
 * Each element in the numerator is divided by the corresponding element in the denominator.
 * It iterates through each pixel, converts `cv::Vec2d` to `std::complex<double>`, performs
 * division, and stores the result back as `cv::Vec2d`.
 *
 * @param numerator The 2-channel complex numerator matrix (e.g., CV_64FC2).
 * @param denominator The 2-channel complex denominator matrix (e.g., CV_64FC2).
 * @return cv::Mat A new 2-channel CV_64FC2 matrix containing the result of the division.
 * @pre numerator and denominator must have the same size and be of type CV_64FC2.
 * @warning Division by zero (or near-zero) in the denominator will result in NaNs or Infs.
 */
cv::Mat divideComplex(const cv::Mat& numerator, const cv::Mat& denominator);

/**
 * @brief Generates a 2D circularly symmetric Gaussian window.
 *
 * This window can be used for smoothing or apodization in the frequency domain.
 * The sigma parameter is fixed relative to the size (size / 4.0f).
 *
 * @param size The side length of the square Gaussian window.
 * @return cv::Mat A single-channel CV_32F matrix representing the Gaussian window, normalized to 1.0 at the center.
 */
cv::Mat gaussianWindow(int size);

/**
 * @brief Computes the Huber function (or a similar robust penalty function) for a given value.
 *
 * This function acts as a robust penalty, which is quadratic for small values (within `sigma`)
 * and linear for large values (outside `sigma`). This helps to reduce the influence of outliers.
 * It's often used in optimization problems, especially those involving L1-like regularization,
 * to provide a smoother, differentiable approximation of the L1 norm while being less sensitive
 * to large errors than an L2 (quadratic) norm.
 *
 * @param x The input value for which to compute the penalty.
 * @return float The computed penalty value.
 */
float phi(float x);

 /**
  * @brief Computes the derivative of the Huber function (or similar robust penalty function).
  *
  * This function provides the derivative of the `phi` function. For values within `sigma`,
  * the derivative is linear; for values outside `sigma`, it becomes a constant (either 1.0 or -1.0),
  * indicating the slope of the linear part of the `phi` function. This derivative is crucial
  * for optimization algorithms that use gradient descent or similar methods.
  *
  * @param x The input value for which to compute the derivative.
  * @return float The computed derivative value.
  */
float dphi(float x); 

// --- Debug helpers ---

/**
 * @brief Writes an OpenCV matrix to a YAML/XML file.
 *
 * This utility function helps in saving the content of an OpenCV matrix to a file,
 * which can be useful for debugging and inspecting intermediate matrix states.
 *
 * @param m The OpenCV matrix to write.
 * @param filename The path and name of the output file (e.g., "matrix_data.yml").
 * @param matrixName The name to assign to the matrix within the file (e.g., "my_matrix").
 */
void writeMatrix(const cv::Mat& m, const std::string& filename, const std::string& matrixName);

/**
 * @brief Checks an OpenCV matrix for NaN (Not a Number) and Inf (Infinity) values.
 *
 * Iterates through the matrix elements (supports 1-channel real and 2-channel complex
 * matrices of CV_32F or CV_64F depth) and reports the count of NaN and Inf values.
 *
 * @param m The OpenCV matrix to check.
 * @param matrixName The name of the matrix for logging purposes.
 */
void checkInfNan(const cv::Mat& m, const std::string& matrixName);

/**
 * @brief Checks a 2-channel complex OpenCV matrix for elements that are exactly zero (0.0 + 0.0i).
 *
 * Reports the count of such elements, which can be useful for debugging denominators in FFT-based operations.
 *
 * @param m The 2-channel complex OpenCV matrix to check (expected CV_64FC2).
 */
void checkExactZero(const cv::Mat& m);

/**
 * @brief Prints statistical information about an OpenCV matrix.
 *
 * Provides min, max, mean, and standard deviation for single-channel matrices.
 * For 2-channel complex matrices, it provides these statistics for both real and imaginary parts.
 *
 * @param mat The OpenCV matrix to analyze.
 * @param name The name of the matrix for logging purposes.
 */
void printMatStats(const cv::Mat& mat, const std::string& name);

/**
 * @brief Combines several debug checks for a given matrix.
 *
 * Calls `writeMatrix`, `checkInfNan`, `checkExactZero`, and `printMatStats`
 * for comprehensive debugging of an OpenCV matrix.
 *
 * @param m The OpenCV matrix to debug.
 * @param matrixName The name of the matrix, used for filenames and logging.
 */
void debugMatrix(const cv::Mat& m, const std::string& matrixName);

/**
 * @brief Generates a 2D Gaussian kernel and normalizes its sum to 1.
 *
 * @param size The side length of the square kernel.
 * @param sigma The standard deviation of the Gaussian function.
 * @return cv::Mat A single-channel CV_32F matrix representing the normalized Gaussian kernel.
 */
cv::Mat createGaussianKernel2D(int size, double sigma);

/**
 * @brief Generates a synthetic blurred image and the corresponding blur kernel.
 *
 * Loads an input grayscale image, applies a Gaussian blur, and saves
 * both the blurred image and the Gaussian kernel as image files.
 * The blurred image is saved scaled to 0-255. The kernel is scaled for visualization.
 *
 * @param input_path Path to the input grayscale image.
 * @param output_blur_path Path to save the generated blurred image.
 * @param output_kernel_path Path to save the visualization of the blur kernel.
 */
void generateSyntheticBlur(const std::string& input_path, const std::string& output_blur_path, const std::string& output_kernel_path);