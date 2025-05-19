/**
 * @file
 * @brief ASCII Art - Transform images into ASCII art!
 * @author Erwin Meza Vega <emezav@gmail.com>
 * @copyright MIT License
 */

 #ifndef ASCII_ART_H
 #define ASCII_ART_H

 #include <iostream>
#include <tuple>
#include <string>

#include <ImagesCPU.h>
#include <ImagesNPP.h>

using std::tuple;
using std::string;
using std::ostream;


/**
 * @brief Prints program usage
 * @param program executable path
 */
void usage(char * program);

/**
 * Sets up a NPP Stream Context
 * @param nppStreamCtx Reference to the NppStreamContext to configure
 * @param Stream to be associated to the context (null stream by default)
 * @return NPP_SUCCESS if successful, some NPP error otherwise.
 */
NppStatus getStreamContext(NppStreamContext &nppStreamCtx, cudaStream_t stream = 0);

/**
 * @brief Loads a 8-bit single channel image into host and device
 * @param imagePath Path to the image file
 * @param hostImage Reference to the destination host image
 * @param deviceImage Reference to the destination device image
 * @return true if the image is found and loaded into host and device, false otherwise
 */
bool getCPUandDeviceImage(const string &imagePath, npp::ImageCPU_8u_C1 &hostImage, npp::ImageNPP_8u_C1 &deviceImage);

/**
 * @brief Resizes an 8-bit single channel image
 * @param src Source image on device
 * @param dstSize Destination image size
 * @param dst Destination image reference
 * @param nppStreamCtx Stream context (required on 12.9)
 */
NppStatus resizeDeviceImage(npp::ImageNPP_8u_C1 &src, NppiSize dstSize, npp::ImageNPP_8u_C1 &dst, const NppStreamContext &nppStreamCtx);

/**
 * @brief Sends an ASCII representation of the device image to a stream
 * @param out Output stream to send the ASCII representation
 * @param img Device image
 * @param filter Edge detection filter
 * @param ASCII pattern to interpret grey intensity. [0] is black, [.length() - 1] is white.
 * @return Reference to the updated output stream
 */
ostream &outAsciiArt(ostream &out, npp::ImageNPP_8u_C1 &img, int filter = -1, string asciiPattern= "");

/**
 * @brief Apply a convolution filter to the source image on device
 * @param src Source image on device
 * @param dst Destination image on device
 * @param kernel Convolution kernel
 * @param kernelSize Convolution kernel size
 * @param anchor Filter starting position (anchor)
 * @param divisor Filter divisor
 * @param nppStreamCtx Stream context (required on 12.9)
 */
NppStatus convolutionFilter(npp::ImageNPP_8u_C1 &src,
                            npp::ImageNPP_8u_C1 &dst,
                            Npp32s *kernel,
                            NppiSize kernelSize,
                            NppiPoint anchor,
                            Npp32s divisor,
                            const NppStreamContext &nppStreamCtx);




#endif

