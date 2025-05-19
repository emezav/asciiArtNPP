/**
 * @file
 * @brief ASCII Art - Transform images into ASCII art!
 * Based on Samples/4_CUDA_Libraries/boxFilterNPP from CUDA Samples
 * Common/ folder code copied from NVIDIA CUDA samples v 12.9 https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v12.9.zip
 * See README.md for more details.
 * @author Erwin Meza Vega <emezav@gmail.com>
 * @copyright MIT License
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <cmath>
#include <cuda_runtime.h>
#include <filesystem> // Requires c++ 17
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>

#include <helper_cuda.h>
#include <helper_string.h>
#include <npp.h>
#include <string.h>

#include "ascii_art.h"

using namespace std;
namespace fs = std::filesystem;

void usage(char * program) {
  cout
  << "ASCII Art - PGM to ASCII Art." << endl
  << "  Usage: " << program << "image.pgm [width [filter [asciiPattern]]]" << endl
  << "  Applies one of the edge detection filters over the input image" << endl
  << "  width: Width of the ASCII representation, 0 = original size, default = 80" << endl
  << "  asciiPattern: ASCII pattern to calculate gray scale. First character is black, last is white." << endl
  << "  - 1 : Sobel X" << endl
  << "  - 2 : Sobel Y" << endl
  << "  - 3 : Scharr X" << endl
  << "  - 4 : Scharr Y" << endl
  << "  - 5 : Scharr X improved" << endl
  << "  - 6 : Scharr Y improved" << endl
  << "  - 7 : Kayali X" << endl
  << "  - 9 : Kayali Y" << endl
  << "  - 9 : Prewitt X" << endl
  << "  - 10: Prewitt Y" << endl;
}

bool getCPUandDeviceImage(const string &imagePath, npp::ImageCPU_8u_C1 &hostImage, npp::ImageNPP_8u_C1 &deviceImage)
{
    try
    {
        // Load image on host
        npp::ImageCPU_8u_C1 oHost;
        npp::loadImage(imagePath, oHost);
        // Create image on device. This allocates memory and copies to device.
        npp::ImageNPP_8u_C1 oDevice(oHost);
        // Update target reference to host
        oHost.swap(hostImage);
        // Update target reference to device
        oDevice.swap(deviceImage);
    }
    catch (exception &e)
    {
        cerr << e.what();
        return false;
    }
    return true;
}

NppStatus resizeDeviceImage(npp::ImageNPP_8u_C1 &src, NppiSize dstSize, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{

    // Get source image dimensions
    NppiSize srcSize = {(int)src.width(), (int)src.height()};

    // Source ROI - whole image {upper left x, upper left y, ROI width, ROI height}
    NppiRect srcROI = {0, 0, srcSize.width, srcSize.height};

    // allocate device image of appropriate size
    npp::ImageNPP_8u_C1 deviceDst(dstSize.width, dstSize.height);

    // output ROI {upper left x, upper left y, ROI width, ROI height}
    NppiRect dstROI = {0, 0, dstSize.width, dstSize.height};

    int eInterpolation = NPPI_INTER_CUBIC;

    NppStatus nppStatus = nppiResize_8u_C1R_Ctx(
        src.data(),
        src.pitch(),
        srcSize,
        srcROI,
        deviceDst.data(),
        deviceDst.pitch(),
        dstSize,
        dstROI,
        eInterpolation,
        nppStreamCtx);

    // Save result to dst
    dst.swap(deviceDst);

    // Return operation status
    return nppStatus;
}

NppStatus getStreamContext(NppStreamContext &nppStreamCtx, cudaStream_t stream)
{

    cudaError_t cudaError;

    // Get CUDA device Id
    cudaError = cudaGetDevice(&nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
    {
        cerr << "CUDA error: no devices supporting CUDA." << endl;
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;
    }

    // Associate this stream context to the selected stream
    nppStreamCtx.hStream = stream;

    // Get NPP version
    const NppLibraryVersion *libVer = nppGetLibVersion();

    // cout << "NPP Library Version" << libVer->major << "." << libVer->minor << "." << libVer->build << endl;

    // Get driver and runtime version
    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    // cout << "CUDA Driver  Version: " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << endl;
    // cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << endl;

    // Set capability fields on this stream context
    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMajor,
                                       cudaDevAttrComputeCapabilityMajor,
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaDeviceGetAttribute(&nppStreamCtx.nCudaDevAttrComputeCapabilityMinor,
                                       cudaDevAttrComputeCapabilityMinor,
                                       nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    cudaError = cudaStreamGetFlags(nppStreamCtx.hStream, &nppStreamCtx.nStreamFlags);

    cudaDeviceProp oDeviceProperties;

    // Set properties
    cudaError = cudaGetDeviceProperties(&oDeviceProperties, nppStreamCtx.nCudaDeviceId);
    if (cudaError != cudaSuccess)
        return NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY;

    nppStreamCtx.nMultiProcessorCount = oDeviceProperties.multiProcessorCount;
    nppStreamCtx.nMaxThreadsPerMultiProcessor = oDeviceProperties.maxThreadsPerMultiProcessor;
    nppStreamCtx.nMaxThreadsPerBlock = oDeviceProperties.maxThreadsPerBlock;
    nppStreamCtx.nSharedMemPerBlock = oDeviceProperties.sharedMemPerBlock;

    return NPP_SUCCESS;
}

ostream &outAsciiArt(ostream &out, npp::ImageNPP_8u_C1 &img,  int filter, string asciiPattern)
{

    if (!asciiPattern.length())
    {
        asciiPattern = "  -.,-=+:;cba?0123456789$WN#@";
        // asciiPattern ="    .:-i|=+xO#@";
    }

    // Create host image based on device image size
    npp::ImageCPU_8u_C1 hostImg(img.size());

    // Get image size
    NppiSize imgSize = {(int)hostImg.width(), (int)hostImg.height()};

    // Copy to host
    img.copyTo(hostImg.data(), hostImg.pitch());

    int patternLength = asciiPattern.length();
    for (int i = 0; i < imgSize.height; i++)
    {
        for (int j = 0; j < imgSize.width; j++)
        {
            int grey = hostImg.pixels(j, i)->x;
            int patternIndex = (grey * patternLength - 1) / 255;
            out << asciiPattern[patternIndex];
        }
        out << endl;
    }
    return out;
}

NppStatus convolutionFilter(npp::ImageNPP_8u_C1 &src,
                            npp::ImageNPP_8u_C1 &dst,
                            Npp32s *kernel,
                            NppiSize kernelSize,
                            NppiPoint anchor,
                            Npp32s divisor,
                            const NppStreamContext &nppStreamCtx)
{
    cudaError_t err;

    // Calculate source ROI
    NppiSize srcROI = {(int)src.width() - kernelSize.width + 1, (int)src.height() - kernelSize.height + 1};
    // Allocate device memory for the output image
    npp::ImageNPP_8u_C1 deviceDst(srcROI.width, srcROI.height); // allocate device image of appropriately reduced size
    Npp32s *deviceKernel;
    // Allocate memory for the kernel and copy to device
    cudaMalloc((void **)&deviceKernel, kernelSize.width * kernelSize.height * sizeof(Npp32s));
    cudaMemcpy(deviceKernel, kernel, kernelSize.width * kernelSize.height * sizeof(Npp32s), cudaMemcpyHostToDevice);

    // Apply convolution filter
    NppStatus nppStatus = nppiFilter_8u_C1R_Ctx(src.data(), src.pitch(),
                                                deviceDst.data(), deviceDst.pitch(),
                                                srcROI, deviceKernel, kernelSize, anchor, divisor, nppStreamCtx);

    // Release device kernel memory
    err = cudaFree(deviceKernel);

    if (err != cudaSuccess)
    {
        return NPP_MEMCPY_ERROR;
    }

    // Store result into destination reference
    deviceDst.swap(dst);

    return NPP_NO_ERROR;
}


/**
 * @brief Available filters. Feel free to add more!
 */
typedef enum {
    SOBEL_X,
    SOBEL_Y,
    SCHARR_X,
    SCHARR_Y,
    SCHARR_X_IMPROVED,
    SCHARR_Y_IMPROVED,
    KAYALI_X,
    KAYALI_Y,
    PREWITT_X,
    PREWITT_Y
}ConvolutionFilter;

/**
 * @brief Sobel X filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus SobelXFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Sobel x
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Sobel Y filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus SobelYFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Sobel y
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Scharr X filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus ScharrXFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {3, 0, -3, 10, 0, -10, 3, 0, -3}; // Scharr x
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Scharr Y filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus ScharrYFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {3, 10, 3, 0, 0, 0, -3, -10, -3}; // Scharr y
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Scharr X filter (improved)
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus ScharrXImprovedFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {47, 0, -47, 162, 0, -162, 47, 0, -47}; // Scharr x (2)
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Scharr Y filter (improved)
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus ScharrYImprovedFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {47, 162, 47, 0, 0, 0, -47, -162, -47}; // Scharr y (2)
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Kayali X filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus KayaliXFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {6, 0, -6, 0, 0, 0, -6, 0, 6}; // Kayali x
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Kayali Y filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus KayaliYFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {-6, 0, 6, 0, 0, 0, 6, 0, -6}; // Kayali y
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Prewitt X filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus PrewittXFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {1, 1, 1, 0, 0, 0, -1, -1, -1}; // Prewitt X
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Prewitt Y filter
 * @param src Source image on device
 * @param dst Destination image to store result
 * @param nppStreamCtx Stream context (required on 12.9)
 * @return Operation status (success = NPP_NO_ERROR)
 */
NppStatus PrewittYFilter(npp::ImageNPP_8u_C1 &src, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx)
{
    Npp32s kernel[9] = {1, 0, -1, 1, 0, -1, 1, 0, -1}; // Prewitt Y
    return convolutionFilter(src, dst, kernel, {3, 3}, {2, 2}, 1, nppStreamCtx);
}

/**
 * @brief Applies the selected convolution filter
 * @param filter number
 * @param src Source image on device
 * @param dst Reference to destination image where result is stored
 * @preturn NPP_NO_ERROR on success, npp error otherwise.
 */
NppStatus applyConvolutionFilter(int filter,
    npp::ImageNPP_8u_C1 &src,
    npp::ImageNPP_8u_C1 &dst,
    const NppStreamContext &nppStreamCtx)
{
    switch (filter) {
        case SOBEL_X: {
            return SobelXFilter(src, dst, nppStreamCtx);
        }
        case SOBEL_Y: {
            return SobelYFilter(src, dst, nppStreamCtx);
        }
        case SCHARR_X: {
            return ScharrXFilter(src, dst, nppStreamCtx);
        }
        case SCHARR_Y: {
            return ScharrYFilter(src, dst, nppStreamCtx);
        }
        case SCHARR_X_IMPROVED: {
            return ScharrXImprovedFilter(src, dst, nppStreamCtx);
        }
        case SCHARR_Y_IMPROVED: {
            return ScharrYImprovedFilter(src, dst, nppStreamCtx);
        }
        case KAYALI_X: {
            return KayaliXFilter(src, dst, nppStreamCtx);
        }
        case KAYALI_Y: {
            return KayaliYFilter(src, dst, nppStreamCtx);
        }
        case PREWITT_X: {
            return PrewittXFilter(src, dst, nppStreamCtx);
        }
        case PREWITT_Y: {
            return PrewittYFilter(src, dst, nppStreamCtx);
        }
        default: {
            return PrewittXFilter(src, dst, nppStreamCtx);
        }
    }
}

/**
 * @brief Image ASCII Art. Transforms an 8-bit gray image to ASCII art
 * @param imagePath Image path
 * @param outColumns Width of the ASCII art, defaults to 80. 0 = no resize, outColumns < 0: Resize to abs(outColumns)
 * @param filter Edge detection filter
 * @param asciiPattern ASCII pattern to interpret grey intensity. [0] is black, [.length() - 1] is white.
 * @return true if successful, false otherwise.
 */
bool imageASCIIArt(const string &imagePath, int outColumns = 80, int filter=-1, string asciiPattern = "")
{
    fs::path srcPath(imagePath);

    if (!fs::exists(srcPath))
    {
        cerr << "Image " << imagePath << " does not exist or is not accessible" << endl;
        return false;
    }

    cudaError_t err;
    NppStatus nppStatus;

    NppStreamContext nppStreamCtx;

    // Get stream context
    nppStatus = getStreamContext(nppStreamCtx);

    if (nppStatus != NPP_SUCCESS)
    {
        cerr << "Unable to get NPP stream context";
        return false;
    }

    try
    {
        npp::ImageCPU_8u_C1 oHostSrc;
        npp::ImageNPP_8u_C1 oDeviceSrc;

        // Load image into CPU and GPU instances
        getCPUandDeviceImage(imagePath, oHostSrc, oDeviceSrc);

        npp::ImageNPP_8u_C1 oDeviceDst;
        //nppStatus = PrewittXFilter(oDeviceSrc, oDeviceDst, nppStreamCtx);
        nppStatus = applyConvolutionFilter(filter, oDeviceSrc, oDeviceDst, nppStreamCtx);
        if (nppStatus != NPP_NO_ERROR)
        {
            cerr << "Error applying filter" << endl;
            nppiFree(oDeviceSrc.data());
            nppiFree(oDeviceDst.data());
            return false;
        }

        // Calculate outColumns depending on parameters

        // Get size of source image
        NppiSize oSrcSize = {(int)oHostSrc.width(), (int)oHostSrc.height()};

        // cout << "Dimensions: " << oSrcSize.width << " x " << oSrcSize.height << " pitch: " << oHostSrc.pitch() << endl;

        // If outColumns < 0, set to abs(outColumns)
        if (outColumns < 0)
        {
            outColumns = abs(outColumns);
        }
        else if (outColumns == 0)
        {
            outColumns = oSrcSize.width;
        }

        if (outColumns == oSrcSize.width)
        {
            // Don't resize image
            // Create ASCII art and store it into oss
            ostringstream oss;

            outAsciiArt(oss, oDeviceDst, filter, asciiPattern);

            // Send oss to cout, or any other output stream
            cout << oss.str();
        }
        else
        {
            // Resize filtered device image

            npp::ImageNPP_8u_C1 oDeviceDstResized;

            // Get size of the host destination image
            NppiSize oDstSize = {(int)oDeviceDst.width(), (int)oDeviceDst.height()};
            // cout << "Destination size: " << oDstSize.width << " x " << oDstSize.height << endl;

            // Resize resulting image
            // Calculate resize factor to fit into outColumns
            float resizeFactor = (float)outColumns / (float)oSrcSize.width;

            NppiSize oDstResizedSize = oDstSize;

            if (resizeFactor < 1)
            {
                oDstResizedSize = {(int)ceil((float)oSrcSize.width * resizeFactor), (int)ceil((float)oSrcSize.height * resizeFactor)};
            }

            // Resize te image and store on oDstResizedSize
            nppStatus = resizeDeviceImage(oDeviceDst, oDstResizedSize, oDeviceDstResized, nppStreamCtx);

            if (nppStatus != NPP_NO_ERROR)
            {
                cerr << "Error resizing image" << endl;
                return false;
            }

            // Create ASCII art and store it into oss
            ostringstream oss;

            outAsciiArt(oss, oDeviceDstResized, filter, asciiPattern);

            // Send oss to cout, or any other output stream
            cout << oss.str();

            // Release resized image data
            nppiFree(oDeviceDstResized.data());
        }

        // Free nppi device memory
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());
    }
    catch (npp::Exception &ex)
    {
        cerr << ex.message() << endl;
        return false;
    }

    return true;
}

int main(int argc, char *argv[])
{

    // Path to a pgm image
    string imagePath = "teapot512.pgm";

    // Default - resize to 80 chars width terminal, set to 0 for no resize
    int columnWidth = 80;

    // ASCII pattern. It should start with space character " " for black.
    string asciiPattern = "  -.,-=+:;cba?0123456789$WN#@";

    // Edge detection filter.
    int filter = -1;

    // Parse image path
    if (argc == 1)
    {
        usage(argv[0]);
        exit(0);
    }

    // Get image path
    imagePath = string(argv[1]);

    // Parse column width
    if (argc > 2)
    {
        columnWidth = std::stoi(argv[2]);
    }

    if (argc > 3) {
        filter = std::stoi(argv[3]);
    }

    // Parse ASCII pattern
    if (argc > 4)
    {
        asciiPattern = string(argv[4]);
    }

    // Do de magic!
    imageASCIIArt(imagePath, columnWidth, filter, asciiPattern);
}