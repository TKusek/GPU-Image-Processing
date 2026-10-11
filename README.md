# GPU Image Processing

An image processing project implemented in **C++, CUDA, and OpenCL**.

The project currently implements image processing operations on both the CPU and GPU, with the goal of exploring how image processing can be parallelized and accelerated using GPU computing and optimization.

## Current Features

* Grayscale image conversion
* Gaussian blur
* CPU implementation
* CUDA implementation
* OpenCL implementation
* Optimized GPU implementations using GPU-specific memory and work organization

## Technologies

* C++
* CUDA
* OpenCL
* `stb_image` for loading images
* `stb_image_write` for saving images

## Project Structure

```text
GPU-Image-Processing/
├── images/
│   ├── original/
│   └── processed/
│
├── include/
│   ├── stb_image.h
│   └── stb_image_write.h
│
├── results/
│
├── src/
│   ├── benchmarker.py
│   ├── CPUMain.cpp
│   ├── CUDAMain.cu
│   ├── OpenCLKernels.cl
│   ├── OpenCLMain.cpp
│   └── stb.cpp
│
├── .gitignore
├── README.md
└── LICENSE
```

## Purpose

The main purpose of this project is to explore **GPU programming and parallel computing**.

The same image processing operations are implemented using different approaches in order to understand how the work can be distributed across CPU threads and GPU threads/work-items.

The project also serves as an opportunity to better understand concepts such as:

* GPU kernels
* Threads/work-items, blocks/work-groups
* GPU memory management
* CUDA shared memory/OpenCL local memory
* Data transfer between CPU and GPU
* Image processing

## Building

The project is currently built using the CUDA Toolkit, OpenCL, and a C++ compiler.

Build commands and additional setup instructions might be added as the project develops, for simplicity.

## Results

The implementations were benchmarked using a 1920 × 1080 RGB image for a total of 2,073,600 pixels. Each configuration was measured 20 times after 3 warm-up runs. The reported values represent the average processing time measured internally by each program, excluding process startup and other work outside the measured processing section. Kernel setup time and other can be shown by starting each compiled program individually, and may differ after a couple executions.

### Average Processing Time

| Implementation | Operation | Average Time (ms) |
|---|---|---:|
| CPU | Grayscale | 7.088 |
| CUDA | Grayscale | 4.603 |
| OpenCL | Grayscale | 2.180 |
| CPU | Gaussian Blur | 85.034 |
| CUDA | Gaussian Blur | 5.458 |
| CUDA | Optimized Gaussian Blur | 4.984 |
| OpenCL | Gaussian Blur | 6.104 |
| OpenCL | Optimized Gaussian Blur | 5.223 |

### Performance Comparison

- **Grayscale:** CUDA was approximately 1.54× faster than the CPU implementation, while OpenCL was approximately 3.25× faster.
- **Gaussian Blur:** CUDA was approximately 15.58× faster than the CPU implementation, while OpenCL was approximately 13.93× faster.
- **CUDA optimization:** The optimized Gaussian blur implementation reduced average processing time by approximately 8.7% compared to the basic CUDA implementation.
- **OpenCL optimization:** The optimized Gaussian blur implementation reduced average processing time by approximately 14.4% compared to the basic OpenCL implementation.

The results demonstrate the benefits of GPU parallelism for image processing, particularly for computationally more demanding operations such as Gaussian blur. The optimized GPU implementations also achieved additional performance improvements over their respective basic implementations.

These measurements are specific to the tested hardware, image size, and implementations. They should not be interpreted as a general performance ranking of CUDA and OpenCL.

The raw benchmark results are available in [`results/benchmark.csv`](results/benchmark.csv).

## Future Improvements

* Further optimize GPU implementations
* Add additional operations

## Author's notes

The operations used for benchmarking were obviously too simple to make using GPU worth it timewise, as the time requiered for GPU initialization, memory transfers and other overhead always overshadowed the kernel time, making the program overall not much faster or, in some instances, even slower than the CPU-only implementation.

Programming the naive algorithms for the GPU was relatively straightforward, it was simply telling the computer what to do, but in a different way. 
The biggest challenge, apart from setting up the CUDA and OpenCL environments, was switching from a naive approach to the optimized kernel for the Gaussian blur, as that required a different approach to the memory access and a couple of new steps to th ealgorithm logic.
