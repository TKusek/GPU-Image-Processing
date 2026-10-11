#include <iostream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <vector>

#include "../include/stb_image.h"

#include "../include/stb_image_write.h"




void grayscale(int pixelCount, unsigned char* image){
    const auto start = std::chrono::high_resolution_clock::now();

    for(int index = 0; index < pixelCount * 3; index = index + 3){
        const unsigned char r = image[index];
        const unsigned char g = image[index + 1];
        const unsigned char b = image[index + 2];

        const unsigned char gray = static_cast<unsigned char>(
                0.299 * r +
                0.587 * g +
                0.114 * b
            );

        image[index] = gray;
        image[index + 1] = gray;
        image[index + 2] = gray;
    }

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "--GRAYSCALE--\n";
    std::cout << "CPU processing time: " << elapsed.count() << " ms\n";
}


void blur(int pixelCount, int width, unsigned char* image){
    const size_t imageSize = static_cast<size_t>(pixelCount) * 3; 
    std::vector<unsigned char> temp(imageSize);

    std::copy(image, image + imageSize, temp.begin());

    const auto start = std::chrono::high_resolution_clock::now();

    constexpr int kernel[9] = {
        1, 8, 28, 56, 70, 56, 28, 8, 1
    };
    //Horizontal vector
    for(int yIndex = 0; yIndex < pixelCount/width - 4; yIndex++)
    for(int xIndex = 4; xIndex < width -4; xIndex++){
        const int index = (yIndex * width + xIndex) * 3;

        int rSum = 0;
        int gSum = 0;
        int bSum = 0;

        for (int kx = -4; kx <= 4; kx++)
        {
            const int neighborIndex =
                (yIndex * width + (xIndex + kx)) * 3;

            const int weight = kernel[kx + 4];

            rSum += image[neighborIndex] * weight;
            gSum += image[neighborIndex + 1] * weight;
            bSum += image[neighborIndex + 2] * weight;
            
        }

        temp[index]     = rSum / 256;
        temp[index + 1] = gSum / 256;
        temp[index + 2] = bSum / 256;
    }
    //Vertical vector
    for (int yIndex = 4; yIndex < pixelCount/width - 4; yIndex++)
    for (int xIndex = 0; xIndex < width; xIndex++)
    {
        const int index = (yIndex * width + xIndex) * 3;

        int rSum = 0;
        int gSum = 0;
        int bSum = 0;

        for (int ky = -4; ky <= 4; ky++)
        {
            const int neighborIndex =
                ((yIndex + ky) * width + xIndex) * 3;

            const int weight = kernel[ky + 4];

            rSum += temp[neighborIndex] * weight;
            gSum += temp[neighborIndex + 1] * weight;
            bSum += temp[neighborIndex + 2] * weight;
        }

        image[index]     = rSum / 256;
        image[index + 1] = gSum / 256;
        image[index + 2] = bSum / 256;
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "--GAUSSIAN BLUR--\n";   
    std::cout << "CPU processing time: " << elapsed.count() << " ms\n";
}



int main(int argc, char* argv[]){
    if (argc < 2){
        std::cerr << "Usage: CUDAMain.exe <operation>\n";
        return 1;
    }
    std::string operation = argv[1];

    const char* inputPath = "images/original/ColoredDog.png";
    std::string outputPath;

    if (operation == "gray") outputPath = "images/processed/grayscale/ColoredDog_grayscale.png";
    else if (operation == "blur") outputPath = "images/processed/gaussian_blur/ColoredDog_blurred.png";
    else{
        std::cerr << "Unknown operation: " << operation << '\n';
        return 1;
    }

    int width;
    int height;
    int channels;
    unsigned char* image = stbi_load(
        "images/original/ColoredDog.png",
        &width,
        &height,
        &channels,
        0
    );

    if (image == nullptr){
        std::cerr << "Failed to load image.\n";
        return 1;
    }
    std::cout << "Image loaded successfully.\n";
    std::cout << "Width: " << width << "\n";
    std::cout << "Height: " << height << "\n";
    std::cout << "Channels: " << channels << "\n";

    const int pixelCount = width * height;
    
    if (operation == "gray") grayscale(pixelCount, image);
    else blur(pixelCount, width, image);

    if (!stbi_write_png(
            outputPath.c_str(),
            width,
            height,
            channels,
            image,
            width * channels)
        ){
        std::cerr << "Failed to save image.\n";
        
        stbi_image_free(image);
        image = nullptr;

        return 1;
    }

    stbi_image_free(image);
    image = nullptr;

    return 0;
}