#include <iostream>
#include <chrono>

#include "../include/stb_image.h"

#include "../include/stb_image_write.h"




__global__ void grayscaleKernel(unsigned char* image, int pixelCount){
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixelCount) return;

    const int index = pixel * 3;

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

__constant__ int kernelV[9];
__global__ void blurKernelH(unsigned char* image, unsigned char* temp, int pixelCount, int width){
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixelCount) return;

    const int x = pixel % width;
    if (x < 4 || x > width-5) return;
    const int y = pixel / width;
    const int index = y * width + x;
    
    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for(int kx = -4; kx <= 4; kx++){
        const int neighborIndex = (index + kx) * 3;
        const int weight = kernelV[kx + 4];

        rSum += image[neighborIndex] * weight;
        gSum += image[neighborIndex + 1] * weight;
        bSum += image[neighborIndex + 2] * weight;
    }

    temp[index * 3] = rSum / 256;
    temp[index * 3 + 1] = gSum / 256;
    temp[index * 3 + 2] = bSum / 256;
}
__global__ void blurKernelV(unsigned char* temp, unsigned char* image, int pixelCount, int width){
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixelCount) return;

    const int x = pixel % width;
    const int y = pixel / width;
    if (y < 4 || y > pixelCount / width - 5) return;
    const int index = y * width + x;
    
    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for(int ky = -4; ky <= 4; ky++){
        const int neighborIndex = (index + ky * width) * 3;
        const int weight = kernelV[ky + 4];

        rSum += temp[neighborIndex] * weight;
        gSum += temp[neighborIndex + 1] * weight;
        bSum += temp[neighborIndex + 2] * weight;
    }

    image[index * 3] = rSum / 256;
    image[index * 3 + 1] = gSum / 256;
    image[index * 3 + 2] = bSum / 256;
}

__global__ void blurKernelHOptimized(
    const unsigned char* image,
    unsigned char* temp,
    int width){
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    __shared__ unsigned char tile[(256 + 8) * 3];

    const int localX = threadIdx.x + 4; //thread X will change the pixel X+4 using [X, X+8]

    if (x < width){  //Filling tile without the halo, but also the rightmost right halos
        const int globalIndex = (y * width + x) * 3;
        const int localIndex = localX * 3;

        tile[localIndex] = image[globalIndex];
        tile[localIndex + 1] = image[globalIndex + 1];
        tile[localIndex + 2] = image[globalIndex + 2];
    }

    if (threadIdx.x < 4){   //Filling tile in the leftmost elements, th eleft halo, except for the leftmost blocks
        const int haloX = x - 4;
        if (haloX >= 0){
            const int globalIndex = (y * width + haloX) * 3;
            const int localIndex = threadIdx.x * 3;

            tile[localIndex] = image[globalIndex];
            tile[localIndex + 1] = image[globalIndex + 1];
            tile[localIndex + 2] = image[globalIndex + 2];
        }
    }

        //Filling the right halos for all but the last one(already filled)
    if (threadIdx.x >= blockDim.x - 4 && blockIdx.x != gridDim.x - 1){
        const int haloX = x + 4;

        const int globalIndex = (y * width + haloX) * 3;
        const int localIndex = (localX + 4) * 3;

        tile[localIndex]     = image[globalIndex];
        tile[localIndex + 1] = image[globalIndex + 1];
        tile[localIndex + 2] = image[globalIndex + 2];
    }

    __syncthreads();

    if (x < 4 || x >= width - 4) return;

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for (int kx = -4; kx <= 4; kx++){
        const int localIndex = (localX + kx) * 3;
        const int weight = kernelV[kx + 4];

        rSum += tile[localIndex] * weight;
        gSum += tile[localIndex + 1] * weight;
        bSum += tile[localIndex + 2] * weight;
    }

    const int globalIndex = (y * width + x) * 3;

    // temp[globalIndex] = 0;   //For testing singel kernel
    // temp[globalIndex + 1] = 0;
    // temp[globalIndex + 2] = 0;
    temp[globalIndex] = rSum / 256;
    temp[globalIndex + 1] = gSum / 256;
    temp[globalIndex + 2] = bSum / 256;
}
__global__ void blurKernelVOptimized(
    unsigned char* temp,
    unsigned char* image,
    int pixelCount,
    int width){
    const int height = pixelCount / width;

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 4 rows top halo
    // 8 rows main data
    // 4 rows bottom halo

    __shared__ unsigned char tile[16 * 32 * 3]; //Twice the size of th block * channels

    const int localX = threadIdx.x;
    const int localY = threadIdx.y + 4;

    //temp -> tile without halo

    if (x < width && y < height){
        const int globalIndex = (y * width + x) * 3;
        const int localIndex = (localY * blockDim.x + localX) * 3;

        tile[localIndex] = temp[globalIndex];
        tile[localIndex + 1] = temp[globalIndex + 1];
        tile[localIndex + 2] = temp[globalIndex + 2];
    }

    //Top halo

    if (threadIdx.y < 4 && blockIdx.y != 0 && x < width){
        const int haloY = y - 4;

        const int globalIndex = (haloY * width + x) * 3;
        const int localIndex = (threadIdx.y * blockDim.x + localX) * 3;

        tile[localIndex] = temp[globalIndex];
        tile[localIndex + 1] = temp[globalIndex + 1];
        tile[localIndex + 2] = temp[globalIndex + 2];
    }

    //Bottom halo

    if (threadIdx.y >= blockDim.y - 4 && blockIdx.y != gridDim.y - 1 && x < width){
        const int haloY = y + 4;
        const int localHaloY = localY + 4;

        if (haloY < height){
            const int globalIndex = (haloY * width + x) * 3;
            const int localIndex = (localHaloY * blockDim.x + localX) * 3;

            tile[localIndex] = temp[globalIndex];
            tile[localIndex + 1] = temp[globalIndex + 1];
            tile[localIndex + 2] = temp[globalIndex + 2];
        }
    }

    __syncthreads();

    //Border pixels

    if (x >= width ||
        y >= height ||
        y < 4 ||
        y >= height - 4) return;

    //Gaussian blur

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for (int ky = -4; ky <= 4; ky++){
        const int localIndex = ((localY + ky) * blockDim.x + localX) * 3;
        const int weight = kernelV[ky + 4];

        rSum += tile[localIndex] * weight;
        gSum += tile[localIndex + 1] * weight;
        bSum += tile[localIndex + 2] * weight;
    }

    //Writee

    const int globalIndex = (y * width + x) * 3;

    image[globalIndex] = rSum / 256;
    image[globalIndex + 1] = gSum / 256;
    image[globalIndex + 2] = bSum / 256;
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
    else if (operation == "blur" || operation == "blurO") outputPath = "images/processed/gaussian_blur/ColoredDog_blurred.png";
    else{
        std::cerr << "Unknown operation: " << operation << '\n';
        return 1;
    }

    int width;
    int height;
    int channels;
    unsigned char* image = stbi_load(
        inputPath,
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

    //------------CUDA part-------------

    unsigned char* GPUimage = nullptr;    //This pointer exists on RAM but its' value will be a memory location on VRAM
    unsigned char* tempImage = nullptr;
    const size_t imageSize = static_cast<size_t>(pixelCount) * channels * sizeof(unsigned char);
    const dim3 threadsOH(256, 1);
    const dim3 blocksOH(
        (width + threadsOH.x - 1) / threadsOH.x,
        height
    );
    const dim3 threadsOV(32, 8);
    const dim3 blocksOV(
        (width + threadsOV.x - 1) / threadsOV.x,
        (height + threadsOV.y - 1) / threadsOV.y
    );
    const int threadsPerBlock = 256;
    const int blocks = (pixelCount + threadsPerBlock -1) / threadsPerBlock;
    
    const auto start = std::chrono::high_resolution_clock::now();
    
    cudaMalloc(&GPUimage, imageSize);
    if (operation == "blur" || operation == "blurO") cudaMalloc(&tempImage, imageSize);

    const auto mallocEnd = std::chrono::high_resolution_clock::now();

    cudaMemcpy(
        GPUimage,
        image,
        imageSize,
        cudaMemcpyHostToDevice
    );
    if (operation == "blur" || operation == "blurO"){
        int kernelVector[9] = {
            1, 8, 28, 56, 70, 56, 28, 8, 1
        };
        cudaMemcpyToSymbol(
            kernelV,
            kernelVector,
            sizeof(kernelV)
        );
    }
    const auto memcpyEnd = std::chrono::high_resolution_clock::now();

        //CUDA working
    if (operation == "gray")
        grayscaleKernel<<<blocks, threadsPerBlock>>>(
            GPUimage,
            pixelCount
        );
    else if (operation == "blur"){
        blurKernelH<<<blocks, threadsPerBlock>>>(
            GPUimage,
            tempImage,
            pixelCount,
            width
        );
        blurKernelV<<<blocks, threadsPerBlock>>>(
            tempImage,
            GPUimage,
            pixelCount,
            width
        );
    }
    else {
        cudaMemcpy(     //Copying so that the edges are made
            tempImage,
            GPUimage,
            imageSize,
            cudaMemcpyDeviceToDevice
        );

        blurKernelHOptimized<<<blocksOH, threadsOH>>>(
            GPUimage,
            tempImage,
            width
        );
        cudaError_t errorH = cudaGetLastError();
        if (errorH != cudaSuccess){
            std::cerr << "Horizontal optimized kernel error: "
                    << cudaGetErrorString(errorH)
                    << '\n';
            return 1;
        }
        blurKernelVOptimized<<<blocksOV, threadsOV>>>(
            tempImage,
            GPUimage,
            pixelCount,
            width
        );
        cudaError_t errorV = cudaGetLastError();
        if (errorV != cudaSuccess){
            std::cerr << "Vertical optimized kernel error: "
                    << cudaGetErrorString(errorV)
                    << '\n';
            return 1;
        }
    }

    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess){
        std::cerr << "CUDA execution error: "
                << cudaGetErrorString(syncError)
                << '\n';
        return 1;
    }
    
    const auto kernelEnd = std::chrono::high_resolution_clock::now();

    // cudaMemcpy(  //For testing single blur kernel
    //     image,
    //     tempImage,
    //     imageSize,
    //     cudaMemcpyDeviceToHost
    // );
    cudaMemcpy(
        image,
        GPUimage,
        imageSize,
        cudaMemcpyDeviceToHost
    );
    const auto end = std::chrono::high_resolution_clock::now();
    
    const std::chrono::duration<double, std::milli> allocTime = mallocEnd - start;
    const std::chrono::duration<double, std::milli> memcpyToGPUTime = memcpyEnd - mallocEnd;
    const std::chrono::duration<double, std::milli> kernelTime = kernelEnd - memcpyEnd;
    const std::chrono::duration<double, std::milli> memcpyToCPUTime = end - kernelEnd;
    const std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "GPU allocation: " << allocTime.count() << "ms\n";
    std::cout << "CPU -> GPU : " << memcpyToGPUTime.count() << "ms\n";
    std::cout << "Kernel execution : " << kernelTime.count() << "ms\n";
    std::cout << "GPU -> CPU : " << memcpyToCPUTime.count() << "ms\n";
    std::cout << "Total CUDA time: " << duration.count() << "ms\n";
    
    if (!stbi_write_png(
            outputPath.c_str(),
            width,
            height,
            channels,
            image,
            width * channels)
        ){
        std::cerr << "Failed to save image.\n";

        cudaFree(GPUimage);
        GPUimage = nullptr;
        if (tempImage != nullptr){
            cudaFree(tempImage);
            tempImage = nullptr;
        }
        stbi_image_free(image);
        image = nullptr;
        
        return 1;
    }

    cudaFree(GPUimage);
    GPUimage = nullptr;
    if (tempImage != nullptr){
        cudaFree(tempImage);
        tempImage = nullptr;
    }
    stbi_image_free(image);
    image = nullptr;  
    
    return 0;
}