# asciiArtNPP

ASCII Art using NVIDIA 2D Image and Signal Processing Performance Primitives - NPP

Erwin Meza Vega <emezav@gmail.com>

Transforms a 8-bit single channel grayscale image into ASCII art

This project was built as the "CUDA at Scale Independent Project" on the Coursera course: CUDA at Scale for the Enterprise.

Common/ folder contains the helper code from the CUDA samples v12.9.

## TL;DR compilation on linux

```sh
make clean build run
```

If everything is OK, Check resulting *_ascii.txt on data/ folder.

Open .ascii.txt file in VS code and see the minimap, or open with a text editor and resize font with the scroll wheel to see
the ASCII art in its full glory.

Usage:

```sh
asciiArtNPP.exe image [width [filter [asciiPattern]]]
```

Program arguments:

- image: PGM image to converto to ASCII art.
- width: Width of the resulting ASCII art. 0 = original image width
- fiter: One of the pre-defined edge detection filters.
- asciiPattern: ASCII string pattern used to transform gray intensity to ASCII.
  First character represents black, last represents white.

## Project description

According to Wikipedia, ASCII Art is a graphic design technique that uses computers for presentation and consists of pictures pieced together from the 95 printable (from a total of 128) characters defined by the ASCII Standard [1]. This technique can be used to show images on text based devices, such as terminals or embedded devices (or just for fun!).

This project takes the code from Samples/4_CUDA_Libraries/boxFilterNPP.

The program performs the following logic:

- Parse command line arguments: image file, resulting ASCII art width, edge detection filter and ASCII translation table pattern
- Open the PGM image
- Detect edges using one of the predefined filters
- Resize the image if requested by user.
- Apply the AsciiArt filter (Developed on this project) to convert the image to an ASCII representation
- Send the ASCII representation to an output stream, cout by default.

## Building the project

This project requires the CUDA Toolkit and the FreeImage library installed and configured. Makefile is supplied for Linux systems.

CMake can be used used to automate compilation on Linux or Widows systems.

## Building on the Coursera lab environment

On the coursera lab, only the src/ include/ and Common/ folders are required.
data/ folder is optional, it contains sample images.

Browse to the downloaded folder an issue the make commands similar to the course labs:

```sh
make clean build
make run
```


## Building using VS Code

Open the project folder on VS Code, it should set up CMake automatically, find CUDA and FreeImage on Linux. For Windows systems, see below.

Use the run button on the actions bar (bottom of VS Code), or build manually (see below).

## Creating CMake configuration (Linux/Windows)

### Linux CMake configuration

```sh
mkdir build
cmake -G "Unix Makefiles" .
```

### Windows CMake configuration

NOTE: On Windows, the FindFeeImage.cmake script from cuda_samples, located on the cmake/Modules subdir is useless, as it only searches FreeImage on standard Linux include/lib directories. For this reason, the file cmake/Modules/FindFreeImage.cmake must be renamed or deleted to allow CMake using the default findPackage(FreeImage)function, and locate the FreeImage installed using vcpkg.

- Install Visual Studio 2022
- Install Visual Studio Code
- Install Git
- Install vcpkg

After installing Visual Studio, Visual Studio Code and Git, install vcpkg following the [official documentation](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-powershell).

- It's easier to copy the vcpkg files to C:\ and point VCPKG_ROOT to that location (C:\vcpkg).
- Remember adding %VCPKG_ROOT% to the PATH environment variable.

Open a VS Developer command prompt and install FreeImage via vcpkg:

```cmd
vcpkg install freeimage
```

Go to the project directory and execute the following commands:

```sh
mkdir build
cmake -G "Visual Studio 12 2022" .
```

You can issue the cmake command with no parameters to identify the correct Visual Studio version.

## Compiling the source (any platform)

Build:

```cmd
cd build
cmake --build .
```

Debug version:

```cmd
cd build
cmake --build . --config Debug
```

Release version:

```cmd
cd build
cmake --build . --config Release
```

Clean:

Debug version:

```cmd
cd build
cmake --build . --config Debug --target clean
cmake --build . --config Release --target clean
```

## References

- [1] https://en.wikipedia.org/wiki/ASCII_art