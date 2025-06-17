#include <iostream>
#include <filesystem>
#include <fstream>
#include <thread>
#include <mutex>
#include <chrono>
#include <set>
#include <opencv4/opencv2/opencv.hpp>
#include <gdal/gdal_priv.h>
#include <gdal/ogr_spatialref.h>

namespace fs = std::filesystem;

cv::Mat canvas;
std::mutex canvasMutex;

double pixelSize = 0.2; // meters per pixel
int canvasWidth = 10000, canvasHeight = 10000;
double originX = 0.0, originY = 0.0;
bool canvasInitialized = false;

// Global lat/lon reference for UTM zone calculation
double referenceLat = 0.0, referenceLon = 0.0;

bool exifToGPS(const std::string& path, double& lat, double& lon) {
    std::string cmd = "exiftool -n -GPSLatitude -GPSLongitude \"" + path + "\"";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;

    char buffer[256];
    std::string line;
    bool foundLat = false, foundLon = false;

    while (fgets(buffer, sizeof(buffer), pipe)) {
        line = buffer;

        if (line.find("GPS Latitude") != std::string::npos) {
            try {
                lat = std::stod(line.substr(line.find(":") + 1));
                foundLat = true;
            } catch (...) {
                pclose(pipe);
                return false;
            }
        }

        if (line.find("GPS Longitude") != std::string::npos) {
            try {
                lon = std::stod(line.substr(line.find(":") + 1));
                foundLon = true;
            } catch (...) {
                pclose(pipe);
                return false;
            }
        }
    }

    pclose(pipe);
    return foundLat && foundLon;
}

bool coordsToUTM(double lat, double lon, double& x, double& y) {
    OGRSpatialReference srcSRS, dstSRS;
    srcSRS.SetWellKnownGeogCS("WGS84");
    int utmZone = static_cast<int>((lon + 180) / 6) + 1;
    dstSRS.SetUTM(utmZone, lat >= 0);

    OGRCoordinateTransformation* transform = OGRCreateCoordinateTransformation(&srcSRS, &dstSRS);
    if (!transform) return false;

    x = lon;
    y = lat;
    if (!transform->Transform(1, &x, &y)) return false;

    OCTDestroyCoordinateTransformation(transform);
    return true;
}

void placeImage(const std::string& imagePath) {
    double lat, lon, x, y;

    if (!exifToGPS(imagePath, lat, lon)) {
        std::cerr << "[ERROR] GPS not found: " << imagePath << "\n";
        return;
    }

    if (!coordsToUTM(lat, lon, x, y)) {
        std::cerr << "[ERROR] Failed to convert coordinates.\n";
        return;
    }

    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << imagePath << "\n";
        return;
    }

    std::lock_guard<std::mutex> lock(canvasMutex);

    if (!canvasInitialized) {
        originX = x - canvasWidth * pixelSize / 2;
        originY = y + canvasHeight * pixelSize / 2;
        referenceLat = lat;
        referenceLon = lon;
        canvas = cv::Mat::zeros(canvasHeight, canvasWidth, CV_8UC3);
        canvasInitialized = true;
    }

    int px = static_cast<int>((x - originX) / pixelSize);
    int py = static_cast<int>((originY - y) / pixelSize);

    if (px < 0 || py < 0 || px >= canvas.cols || py >= canvas.rows) {
        std::cerr << "[WARN] Image out of bounds: " << imagePath << "\n";
        return;
    }

    int roiWidth = std::min(img.cols, canvas.cols - px);
    int roiHeight = std::min(img.rows, canvas.rows - py);

    if (roiWidth <= 0 || roiHeight <= 0) return;

    cv::Mat roi = canvas(cv::Rect(px, py, roiWidth, roiHeight));
    img(cv::Rect(0, 0, roiWidth, roiHeight)).copyTo(roi);
    std::cout << "[OK] Placed: " << imagePath << " at " << x << "," << y << "\n";
}

void save_geotiff(const std::string& output_path) {
    std::lock_guard<std::mutex> lock(canvasMutex);
    if (!canvasInitialized) return;

    GDALAllRegister();
    const char* format = "GTiff";
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName(format);
    if (!driver) return;

    char** papszOptions = NULL;
    GDALDataset* dst = driver->Create(output_path.c_str(), canvas.cols, canvas.rows, 3, GDT_Byte, papszOptions);
    if (!dst) return;

    double geoTransform[6] = {
        originX, pixelSize, 0,
        originY, 0, -pixelSize
    };
    dst->SetGeoTransform(geoTransform);

    int utmZone = static_cast<int>((referenceLon + 180) / 6) + 1;
    OGRSpatialReference srs;
    srs.SetUTM(utmZone, referenceLat >= 0);
    srs.SetWellKnownGeogCS("WGS84");

    char* wkt;
    srs.exportToWkt(&wkt);
    dst->SetProjection(wkt);
    CPLFree(wkt);

    // Convert canvas from BGR to RGB
    cv::Mat canvasRGB;
    cv::cvtColor(canvas, canvasRGB, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels(3);
    cv::split(canvasRGB, channels);

    for (int i = 0; i < 3; i++) {
        CPLErr err = dst->GetRasterBand(i + 1)->RasterIO(
            GF_Write,
            0, 0,
            canvas.cols, canvas.rows,
            channels[i].data,
            canvas.cols, canvas.rows,
            GDT_Byte,
            0, 0
        );

        if (err != CE_None) {
            std::cerr << "[ERROR] Failed to write band " << i + 1 << "\n";
        }
    }

    GDALClose(dst);
    std::cout << "[SAVE] Mosaic saved to " << output_path << "\n";
}

int main() {
    std::string watchFolder = "incoming";
    std::string outputPath = "stitched/final_orthophoto.tif";

    std::set<std::string> processed;

    while (true) {
        for (const auto& entry : fs::directory_iterator(watchFolder)) {
            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".JPG") {
                std::string imagePath = entry.path().string();
                if (processed.count(imagePath) == 0) {
                    std::thread t(placeImage, imagePath);
                    t.detach();
                    processed.insert(imagePath);
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(5));
        save_geotiff(outputPath);
    }

    return 0;
}
