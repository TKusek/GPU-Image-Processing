__kernel void grayscaleKernel(
    __global unsigned char* image,
    int pixelCount){
    const int pixel = get_global_id(0);
    if (pixel >= pixelCount) return;

    const int index = pixel * 3;
    const unsigned char r = image[index];
    const unsigned char g = image[index + 1];
    const unsigned char b = image[index + 2];
    
    const unsigned char gray = (unsigned char)(
        0.299 * r +
        0.587 * g +
        0.114 * b
    );

    image[index] = gray;
    image[index + 1] = gray;
    image[index + 2] = gray;
}

__constant int kernelVector[9] = {
    1, 8, 28, 56, 70, 56, 28, 8, 1
};
__kernel void blurKernelH(
        __global const unsigned char* image,
        __global unsigned char* temp,
        int pixelCount,
        int width){
    const int pixel = get_global_id(0);
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
        const int weight = kernelVector[kx + 4];

        rSum += image[neighborIndex] * weight;
        gSum += image[neighborIndex + 1] * weight;
        bSum += image[neighborIndex + 2] * weight;
    }

    temp[index * 3] = rSum / 256;
    temp[index * 3 + 1] = gSum / 256;
    temp[index * 3 + 2] = bSum / 256;
}
__kernel void blurKernelV(
        __global const unsigned char* temp,
        __global unsigned char* image,
        int pixelCount,
        int width){
    const int pixel = get_global_id(0);
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
        const int weight = kernelVector[ky + 4];

        rSum += temp[neighborIndex] * weight;
        gSum += temp[neighborIndex + 1] * weight;
        bSum += temp[neighborIndex + 2] * weight;
    }

    image[index * 3] = rSum / 256;
    image[index * 3 + 1] = gSum / 256;
    image[index * 3 + 2] = bSum / 256;
}

__kernel void blurKernelHOptimized(
        __global const unsigned char* image,
        __global unsigned char* temp,
        int pixelCount,     //I just needed an extra argument, because of main codde 
        int width){
    const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
    const int y = get_group_id(1);

    __local unsigned char tile[(256 + 8) * 3];

    const int localX = get_local_id(0) + 4;

    // Load main pixels
    if (x < width){
        const int globalIndex = (y * width + x) * 3;
        const int localIndex = localX * 3;

        tile[localIndex] = image[globalIndex];
        tile[localIndex + 1] = image[globalIndex + 1];
        tile[localIndex + 2] = image[globalIndex + 2];
    }

    // Left halo
    if (get_local_id(0) < 4){
        const int haloX = x - 4;
        if (haloX >= 0){
            const int globalIndex = (y * width + haloX) * 3;
            const int localIndex = get_local_id(0) * 3;

            tile[localIndex] = image[globalIndex];
            tile[localIndex + 1] = image[globalIndex + 1];
            tile[localIndex + 2] = image[globalIndex + 2];
        }
    }

    // Right halo
    if (get_local_id(0) >= get_local_size(0) - 4 &&
        get_group_id(0) != get_num_groups(0) - 1){
        const int haloX = x + 4;

        const int globalIndex = (y * width + haloX) * 3;
        const int localIndex = (localX + 4) * 3;

        tile[localIndex] = image[globalIndex];
        tile[localIndex + 1] = image[globalIndex + 1];
        tile[localIndex + 2] = image[globalIndex + 2];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x < 4 || x >= width - 4) return;

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for (int kx = -4; kx <= 4; kx++){
        const int localIndex = (localX + kx) * 3;
        const int weight = kernelVector[kx + 4];

        rSum += tile[localIndex] * weight;
        gSum += tile[localIndex + 1] * weight;
        bSum += tile[localIndex + 2] * weight;
    }

    const int globalIndex = (y * width + x) * 3;

    temp[globalIndex] = rSum / 256;
    temp[globalIndex + 1] = gSum / 256;
    temp[globalIndex + 2] = bSum / 256;
}
__kernel void blurKernelVOptimized(
        __global const unsigned char* temp,
        __global unsigned char* image,
        int pixelCount,
        int width){
    const int height = pixelCount / width;

    const int x = get_group_id(0) * get_local_size(0)
                + get_local_id(0);
    const int y = get_group_id(1) * get_local_size(1)
                + get_local_id(1);

    __local unsigned char tile[16 * 32 * 3];

    const int localX = get_local_id(0);
    const int localY = get_local_id(1) + 4;

    // Main pixels
    if (x < width && y < height){
        const int globalIndex = (y * width + x) * 3;
        const int localIndex = (localY * 32 + localX) * 3;

        tile[localIndex] = temp[globalIndex];
        tile[localIndex + 1] = temp[globalIndex + 1];
        tile[localIndex + 2] = temp[globalIndex + 2];
    }

    // Top halo
    if (get_local_id(1) < 4 && get_group_id(1) != 0 && x < width){
        const int haloY = y - 4;
        const int globalIndex = (haloY * width + x) * 3;
        const int localIndex = (get_local_id(1) * 32 + localX) * 3;

        tile[localIndex] = temp[globalIndex];
        tile[localIndex + 1] = temp[globalIndex + 1];
        tile[localIndex + 2] = temp[globalIndex + 2];
    }

    // Bottom halo
    if (get_local_id(1) >= get_local_size(1) - 4 &&
        get_group_id(1) != get_num_groups(1) - 1 &&
        x < width){
        const int haloY = y + 4;

        if (haloY < height){
            const int globalIndex = (haloY * width + x) * 3;
            const int localHaloY = localY + 4;
            const int localIndex = (localHaloY * 32 + localX) * 3;

            tile[localIndex] = temp[globalIndex];
            tile[localIndex + 1] = temp[globalIndex + 1];
            tile[localIndex + 2] = temp[globalIndex + 2];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (x >= width ||
        y >= height ||
        y < 4 ||
        y >= height - 4) return;

    int rSum = 0;
    int gSum = 0;
    int bSum = 0;

    for (int ky = -4; ky <= 4; ky++){
        const int localIndex = ((localY + ky) * 32 + localX) * 3;

        const int weight = kernelVector[ky + 4];

        rSum += tile[localIndex] * weight;
        gSum += tile[localIndex + 1] * weight;
        bSum += tile[localIndex + 2] * weight;
    }

    const int globalIndex = (y * width + x) * 3;

    image[globalIndex] = rSum / 256;
    image[globalIndex + 1] = gSum / 256;
    image[globalIndex + 2] = bSum / 256;
}