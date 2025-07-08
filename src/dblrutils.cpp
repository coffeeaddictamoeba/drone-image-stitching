#include "../include/dblrutils.h"
#include <string>
#include <complex>
#include <limits>


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

void debugMatrix(const cv::Mat& m, const std::string& matrixName) {
    const std::string filename = saveDir + "/" + matrixName + ".yml";
    writeMatrix(m, filename, matrixName);
    checkInfNan(m, matrixName);
    checkExactZero(m);
    checkMagnitude(m);
}

void checkMagnitude(const cv::Mat& m) {
    double minMagnitude = std::numeric_limits<double>::max();
    double maxMagnitude = 0.0;
    for (int r = 0; r < m.rows; ++r) {
        for (int c = 0; c < m.cols; ++c) {
            cv::Vec2d val = m.at<cv::Vec2d>(r, c);
            double magnitude = std::abs(val[0]); // Imaginary part is 0, so just real part's abs value
            if (magnitude < minMagnitude) {
                minMagnitude = magnitude;
            }
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
            }
        }
    }
    std::cout << "Debug: Min magnitude (complex): " << minMagnitude << "\n";
    std::cout << "Debug: Max magnitude (complex): " << maxMagnitude << "\n";
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

cv::Mat divideComplex(cv::Mat& numerator, cv::Mat& denominator) {
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