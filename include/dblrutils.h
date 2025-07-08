#include <opencv4/opencv2/opencv.hpp>

/**
 * @brief Computes the optimal DFT size for an image to ensure efficient FFT.
 * @param size The original size of the image.
 * @return The optimal size for DFT.
 */
cv::Size getOptimalDFTSize(const cv::Size& size);

/**
 * @brief Performs a 2D Forward Fast Fourier Transform (FFT).
 * @param input_real The input real-valued image.
 * @param output_complex The output complex-valued DFT result.
 */
void fft2d(const cv::Mat& input_real, cv::Mat& output_complex);

/**
 * @brief Performs a 2D Inverse Fast Fourier Transform (FFT).
 * @param input_complex The input complex-valued DFT result.
 * @param output_real The output real-valued image.
 */
void ifft2d(const cv::Mat& input_complex, cv::Mat& output_real);

/**
 * @brief Saves given matrix to a specified file.
 * @param m The matrix to observe.
 * @param filename The output filename path (.yml format preferred).
 * @param matrixName The name of the matrix for debugging purposes.
 */
void writeMatrix(const cv::Mat& m, const std::string& filename, const std::string& matrixName);

/**
 * @brief Checks amount of Inf and Nan parameters of matrix (1-channel or 2-channel).
 * @param m The matrix to observe.
 * @param matrixName The name of the matrix for debugging purposes.
 */
void checkInfNan(const cv::Mat& m, const std::string& matrixName);

/**
 * @brief Writes all info about a specified matrix.
 * @param m The matrix to observe.
 * @param matrixName The name of the matrix for debugging purposes.
 */
void debugMatrix(const cv::Mat& m, const std::string& matrixName);

/**
 * @brief Checks min and max values of a specified matrix.
 * @param m The matrix to observe.
 */
void checkMagnitude(const cv::Mat& m);

/**
 * @brief Checks amount of zero values of a specified matrix.
 * @param m The matrix to observe.
 */
void checkExactZero(const cv::Mat& m);

/**
 * @brief Divides two complex matrices.
 * @param numerator Numerator matrix for division.
 * @param denominator Denominator matrix for division.
 * @return The result matrix after complex division.
 */
cv::Mat divideComplex(cv::Mat& numerator, cv::Mat& denominator);