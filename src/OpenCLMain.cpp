#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>

#include <CL/cl.h>

#include "../include/stb_image.h"
#include "../include/stb_image_write.h"







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

    //------------OpenCL part-------------

    cl_int err;
    
    const size_t globalSize = pixelCount;   //How many work-items
    const size_t localSize = 256;   //How many work-items per work-group
                                    //Number of work-groups is automatically calculated

    const size_t localSizeHO[2] = {256, 1};
    const size_t globalSizeHO[2] = {
        ((static_cast<size_t>(width) + 255) / 256) * 256,
        static_cast<size_t>(height)
    };

    const size_t localSizeVO[2] = {32, 8};
    const size_t globalSizeVO[2] = {
        ((static_cast<size_t>(width) + 31) / 32) * 32,
        ((static_cast<size_t>(height) + 7) / 8) * 8
    };

    const auto start = std::chrono::high_resolution_clock::now();
    
        //Find any installed OpenCL platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);

        //Find any GPU device which uses that kind of platform
    cl_device_id device;
    clGetDeviceIDs(
        platform,
        CL_DEVICE_TYPE_GPU,
        1,
        &device,
        nullptr
    );

        //Creates a working environment which logically connects the GPU device, the kernel function, and the queue
    cl_context context = clCreateContext(
        nullptr,
        1,
        &device,
        nullptr,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) std::cerr << "Failed to create context\n";

        //Queue is an ordered list of commands for the GPU device, such as "Do the kernel function"
    cl_command_queue queue = clCreateCommandQueue(
        context,
        device,
        0,
        &err
    );
    if (err != CL_SUCCESS) std::cerr << "Failed to create queue\n";

        //Importing the kernel code from .cl file
    std::ifstream kernelFile("src/OpenCLKernels.cl");
    std::stringstream kernelStream;
    kernelStream << kernelFile.rdbuf();
    std::string kernelSource = kernelStream.str();
    const char* source = kernelSource.c_str();

        //Putting the kernel source code into the context
    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &source,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) std::cerr << "Failed to create program\n";

        //Compiles the program for the GPU and keeps it "hidden" with the program variable
    err = clBuildProgram(
        program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr
    );
    if (err != CL_SUCCESS){
        std::cerr << "Failed to build program\n";

        size_t logSize = 0;
        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &logSize
        );

        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(
            program,
            device,
            CL_PROGRAM_BUILD_LOG,
            logSize,
            buildLog.data(),
            nullptr
        );
        std::cerr << buildLog.data() << '\n';

        return 1;
    }
    
        //Creates kernels which use the functions from the .cl file
    cl_kernel kernel1 = nullptr;
    cl_kernel kernel2 = nullptr;
    if (operation == "blur"){
        kernel1 = clCreateKernel(
            program,
            "blurKernelH",
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create blurH kernel\n";
        kernel2 = clCreateKernel(
            program,
            "blurKernelV",
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create blurV kernel\n";
    }
    else if (operation == "blurO"){
        kernel1 = clCreateKernel(
            program,
            "blurKernelHOptimized",
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create blurHO kernel\n";
        kernel2 = clCreateKernel(
            program,
            "blurKernelVOptimized",
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create blurVOS kernel\n";
    }
    else{
        kernel1 = clCreateKernel(
            program,
            "grayscaleKernel",
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create grayscale kernel\n";
    }

        //Reserves a section of memory of the VRAM
    size_t imageSize = static_cast<size_t>(pixelCount) * channels * sizeof(unsigned char);
    
    cl_mem tempImage = nullptr;
    if (operation == "blur" || operation == "blurO"){
        tempImage = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            imageSize,
            nullptr,
            &err
        );
        if (err != CL_SUCCESS) std::cerr << "Failed to create tempImage buffer\n";
    }
    cl_mem GPUImage = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE,
        imageSize,
        nullptr,
        &err
    );
    if (err != CL_SUCCESS) std::cerr << "Failed to create GPUImage buffer\n";

        //Copies the image from RAM to VRAM (puts the action in the command queue)
    clEnqueueWriteBuffer(
        queue,
        GPUImage,
        CL_TRUE,
        0,
        imageSize,
        image,
        0,
        nullptr,
        nullptr
    );

        //Setting arguments for the kernel
    if (operation == "blur" || operation == "blurO"){
        clSetKernelArg(kernel1, 0, sizeof(cl_mem), &GPUImage);
        clSetKernelArg(kernel1, 1, sizeof(cl_mem), &tempImage);
        clSetKernelArg(kernel1, 2, sizeof(int), &pixelCount);
        clSetKernelArg(kernel1, 3, sizeof(int), &width);
        clSetKernelArg(kernel2, 0, sizeof(cl_mem), &tempImage);
        clSetKernelArg(kernel2, 1, sizeof(cl_mem), &GPUImage);
        clSetKernelArg(kernel2, 2, sizeof(int), &pixelCount);
        clSetKernelArg(kernel2, 3, sizeof(int), &width);
    }
    else{
        clSetKernelArg(kernel1, 0, sizeof(cl_mem), &GPUImage);
        clSetKernelArg(kernel1, 1, sizeof(int), &pixelCount);
    }

        //Wait until the queue is empty
    clFinish(queue);

        //Setting how many threads there will be on the GPU and starting kernel fucntion
    const auto startKernel =
    std::chrono::high_resolution_clock::now();

    if (operation == "blurO"){
        err = clEnqueueCopyBuffer(
            queue,
            GPUImage,
            tempImage,
            0,
            0,
            imageSize,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS){
            std::cerr << "Failed to copy GPUImage to tempImage\n";
            return 1;
        }

        err = clEnqueueNDRangeKernel(
            queue,
            kernel1,
            2,
            nullptr,
            globalSizeHO,
            localSizeHO,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS){
            std::cerr << "Failed to enqueue optimized horizontal kernel\n";
            return 1;
        }

        err = clEnqueueNDRangeKernel(
            queue,
            kernel2,
            2,
            nullptr,
            globalSizeVO,
            localSizeVO,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS){
            std::cerr << "Failed to enqueue optimized vertical kernel\n";
            return 1;
        }
    }
    else{
        err = clEnqueueNDRangeKernel(
            queue,
            kernel1,
            1,
            nullptr,
            &globalSize,
            &localSize,
            0,
            nullptr,
            nullptr
        );
        if (err != CL_SUCCESS){
            std::cerr << "Failed to enqueue kernel1\n";
            return 1;
        }

        if (operation == "blur"){
            err = clEnqueueNDRangeKernel(
                queue,
                kernel2,
                1,
                nullptr,
                &globalSize,
                &localSize,
                0,
                nullptr,
                nullptr
            );
            if (err != CL_SUCCESS){
                std::cerr << "Failed to enqueue kernel2\n";
                return 1;
            }
        }
    }

    clFinish(queue);

    const auto endKernel = std::chrono::high_resolution_clock::now();

        //Copying the new image from VRAM to image on RAM
    clEnqueueReadBuffer(
        queue,
        GPUImage,
        CL_TRUE,
        0,
        imageSize,
        image,
        0,
        nullptr,
        nullptr
    );

        //Frees the memory
    clReleaseMemObject(GPUImage);
    GPUImage = nullptr;
    if (tempImage != nullptr){
        clReleaseMemObject(tempImage);
        tempImage = nullptr;
    }
    clReleaseKernel(kernel1);
    kernel1 = nullptr;
    if (kernel2 != nullptr){
        clReleaseKernel(kernel2);
        kernel2 = nullptr;
    }
    clReleaseProgram(program);
    program = nullptr;
    clReleaseCommandQueue(queue);
    queue = nullptr;
    clReleaseContext(context);
    context = nullptr;
//--------------------------------------------------------------------------------------------------

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    const std::chrono::duration<double, std::milli> kernelTime = endKernel - startKernel;
    
    std::cout << "Kernel time: " << kernelTime.count() << "ms\n";
    std::cout << "Total OpenCL time: " << duration.count() << "ms\n";

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