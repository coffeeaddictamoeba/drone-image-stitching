#include "../include/dblrutils.h"
#include <string>
#include <complex>
#include <limits>


// --- Calculation helpers --- 
cv::Size getOptimalDFTSize(const cv::Size& size) {
    int r = cv::getOptimalDFTSize(size.height);
    int c = cv::getOptimalDFTSize(size.width);
    return cv::Size(c, r);
}

void fft2d(const cv::Mat& input_real, cv::Mat& output_complex) {
    CV_Assert(input_real.channels() == 1);
    cv::Mat planes[] = { input_real.clone(), cv::Mat::zeros(input_real.size(), input_real.type()) };
    cv::Mat complex_input;
    cv::merge(planes, 2, complex_input);
    cv::dft(complex_input, output_complex, cv::DFT_COMPLEX_OUTPUT);
}

void ifft2d(const cv::Mat& input_complex, cv::Mat& output_complex) {
    CV_Assert(input_complex.channels() == 2);
    cv::dft(input_complex, output_complex, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
}

cv::Mat fft2dWithPad(const cv::Mat& src, cv::Size size, const cv::Rect& roi) {
    cv::Mat padded = cv::Mat::zeros(size, CV_32F);
    src.copyTo(padded(roi));
    cv::Mat complex;
    fft2d(padded, complex);
    return complex;
}

cv::Mat conj64F(const cv::Mat& complex32) {
    cv::Mat complex64;
    complex32.convertTo(complex64, CV_64FC2);
    std::vector<cv::Mat> planes(2);
    cv::split(complex64, planes);
    planes[1] *= -1.0;
    cv::merge(planes, complex64);
    return complex64;
}

cv::Mat magSq64F(const cv::Mat& complex32) {
    cv::Mat c64;
    complex32.convertTo(c64, CV_64FC2);
    std::vector<cv::Mat> planes(2);
    cv::split(c64, planes);
    return planes[0].mul(planes[0]) + planes[1].mul(planes[1]);
}

cv::Mat mergeRealImag(const cv::Mat& real, const cv::Mat& imag) {
    std::vector<cv::Mat> planes = {real, imag};
    cv::Mat merged;
    cv::merge(planes, merged);
    return merged;
}

cv::Mat divideComplex(const cv::Mat& numerator, const cv::Mat& denominator) {
    cv::Mat result(numerator.size(), CV_64FC2);
    for (int r = 0; r < numerator.rows; ++r) {
        for (int c = 0; c < numerator.cols; ++c) {
            cv::Vec2d numv = numerator.at<cv::Vec2d>(r, c);
            cv::Vec2d denv = denominator.at<cv::Vec2d>(r, c);
    
            std::complex<double> num_comp(numv[0], numv[1]);
            std::complex<double> den_comp(denv[0], denv[1]); // denv[1] should be 0.0
    
            std::complex<double> result_comp = num_comp / den_comp;
    
            result.at<cv::Vec2d>(r, c) = cv::Vec2d(result_comp.real(), result_comp.imag());
        }
    }
    return result;
}

cv::Mat gaussianWindow(int size) {
    cv::Mat window(size, size, CV_32F);
    float sigma = size / 4.0f;
    int center = size / 2;

    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float dx = x - center;
            float dy = y - center;
            window.at<float>(y, x) = std::exp(-(dx*dx + dy*dy) / (2 * sigma * sigma));
        }
    }
    return window;
}

// --- Debug helpers ---
void writeMatrix(const cv::Mat& m, const std::string& filename, const std::string& matrixName) {
    try {
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened()) {
            std::cerr << "Error: Could not open the file: " << filename << "\n";
            return;
        }
        fs << matrixName << m;
        fs.release();
        std::cout << "Debug: Matrix '" << matrixName << "' written to " << filename << "\n";
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error writing matrix to file: " << e.what() << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Standard exception writing matrix to file: " << e.what() << "\n";
    }
}

void checkInfNan(const cv::Mat& m, const std::string& matrixName) {
    int nan_count = 0;
    int inf_count = 0;

    if (m.channels() == 2) { // Handle 2-channel complex matrices (e.g., CV_32FC2, CV_64FC2)
        if (m.depth() == CV_32F) {
            for(int r = 0; r < m.rows; ++r) {
                for(int c = 0; c < m.cols; ++c) {
                    cv::Vec2f val = m.at<cv::Vec2f>(r, c);
                    if (std::isnan(val[0]) || std::isnan(val[1])) nan_count++;
                    if (std::isinf(val[0]) || std::isinf(val[1])) inf_count++;
                }
            }
        } else if (m.depth() == CV_64F) {
            for(int r = 0; r < m.rows; ++r) {
                for(int c = 0; c < m.cols; ++c) {
                    cv::Vec2d val = m.at<cv::Vec2d>(r, c);
                    if (std::isnan(val[0]) || std::isnan(val[1])) nan_count++;
                    if (std::isinf(val[0]) || std::isinf(val[1])) inf_count++;
                }
            }
        } else {
            std::cerr << "Error: Unsupported complex matrix depth. Depth: " << m.depth() << "\n";
            return;
        }
    } else if (m.channels() == 1) { // Handle 1-channel real matrices (e.g., CV_32F, CV_64F)
        if (m.depth() == CV_32F) {
            for(int r = 0; r < m.rows; ++r) {
                for(int c = 0; c < m.cols; ++c) {
                    float val = m.at<float>(r, c);
                    if (std::isnan(val)) nan_count++;
                    if (std::isinf(val)) inf_count++;
                }
            }
        } else if (m.depth() == CV_64F) {
            for(int r = 0; r < m.rows; ++r) {
                for(int c = 0; c < m.cols; ++c) {
                    double val = m.at<double>(r, c);
                    if (std::isnan(val)) nan_count++;
                    if (std::isinf(val)) inf_count++;
                }
            }
        } else {
            std::cerr << "Error: Unsupported real matrix depth. Depth: " << m.depth() << "\n";
            return;
        }
    } else {
        std::cerr << "Error: checkInfNan expects 1-channel or 2-channel matrix. Channels: " << m.channels() << "\n";
        return;
    }
    std::cout << "[" << matrixName << "] NaN count: " << nan_count << ", Inf count: " << inf_count << '\n';
}

void checkExactZero(const cv::Mat& m) {
    int exactZeroCount = 0;
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            cv::Vec2d val = m.at<cv::Vec2d>(r, c);
            if (val[0] == 0.0 && val[1] == 0.0) {
                exactZeroCount++;
            }
        }
    }
    std::cout << "Debug: Exact zero complex elements: " << exactZeroCount << "\n";
}

void printMatStats(const cv::Mat& mat, const std::string& name) {
    if (mat.empty()) {
        std::cout << "DEBUG_STATS: " << name << " is empty.\n";
        return;
    }
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    cv::Scalar meanVal, stddevVal;
    cv::meanStdDev(mat, meanVal, stddevVal);

    if (mat.channels() == 1) {
        std::cout << "DEBUG_STATS: " << name << " (1-ch) - Min: " << minVal << ", Max: " << maxVal
                  << ", Mean: " << meanVal[0] << ", StdDev: " << stddevVal[0] << "\n";
    } else if (mat.channels() == 2) { // Complex numbers (CV_64FC2, etc.)
        cv::Mat planes[2];
        cv::split(mat, planes);
        double minReal, maxReal, minImag, maxImag;
        cv::minMaxLoc(planes[0], &minReal, &maxReal);
        cv::minMaxLoc(planes[1], &minImag, &maxImag);
        std::cout << "DEBUG_STATS: " << name << " (2-ch complex) - MinReal: " << minReal << ", MaxReal: " << maxReal
                  << ", MinImag: " << minImag << ", MaxImag: " << maxImag
                  << ", MeanReal: " << meanVal[0] << ", StdDevReal: " << stddevVal[0]
                  << ", MeanImag: " << meanVal[1] << ", StdDevImag: " << stddevVal[1] << "\n";
    } else {
        std::cout << "DEBUG_STATS: " << name << " - Unknown channel count: " << mat.channels() << "\n";
    }
}

void debugMatrix(const cv::Mat& m, const std::string& matrixName) {
    const std::string filename = matrixName + ".yml";
    writeMatrix(m, filename, matrixName);
    checkInfNan(m, matrixName);
    checkExactZero(m);
    printMatStats(m, matrixName);
}

// Generate normalized Gaussian kernel
cv::Mat createGaussianKernel2D(int size, double sigma) {
    cv::Mat kernel1D = cv::getGaussianKernel(size, sigma, CV_32F);
    cv::Mat kernel2D = kernel1D * kernel1D.t();
    return kernel2D / cv::sum(kernel2D)[0];
}

// Apply known blur to a grayscale image
void generateSyntheticBlur(const std::string& input_path, const std::string& output_blur_path, const std::string& output_kernel_path) {
    cv::Mat img = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return;
    }
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    int kernel_size = 25;
    double sigma = 3.0;

    cv::Mat kernel = createGaussianKernel2D(kernel_size, sigma);
    cv::Mat blurred;

    // Apply blur with BORDER_REPLICATE to simulate real blur
    cv::filter2D(img, blurred, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

    // Save results
    cv::imwrite(output_blur_path, blurred * 255);
    double minVal, maxVal;
    cv::minMaxLoc(kernel, &minVal, &maxVal);

    cv::Mat kernel_visual;
    kernel.convertTo(kernel_visual, CV_32F);
    cv::imwrite(output_kernel_path, kernel_visual / maxVal * 255);

    std::cout << "Synthetic blurred image and kernel saved.\n";
}