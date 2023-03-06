#ifdef __clang__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda.h>

#include "cbmp.h"
#include "imageGPU.h"

// Borne une valeur entre deux bornes
#define CLAMP(value, low, high)                                      \
    (value < low ? low : (value > high ? high : value))

// Quitte le programme et affiche un message s'il y a une erreur CUDA
#define EXPECT(err, msg, ...)                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "Cuda error at %s:%d, " msg "\n", __FILE__,  \
                __LINE__, ##__VA_ARGS__);                            \
        fprintf(stderr, " %d: %s \n", err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                          \
    }

// Première communication avec CUDA
// Faite en amont afin d'initialiser le runtime CUDA avant d'effectuer
// les mesures qui nous intéressent
void echauffement() {
    cudaError_t err = cudaSuccess;
    cudaEvent_t start;

    err = cudaEventCreate(&start);
    EXPECT(err, "Couln't create start event");
}

// le noyau qui sera execute sur le GPU
__global__ void niveauDeGrisKernel(pixel *grayImage, pixel *rgbImage,
                                   int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < height && col < width) {
        // Lecture
        int offset = width * row + col;
        pixel pix = rgbImage[offset];

        // Calcul
        // Y = 0.299 R + 0.587 G + 0.114 B
        unsigned char y =
            0.299 * pix.red + 0.587 * pix.green + 0.114 * pix.blue;

        // Ecriture
        grayImage[offset] = pixel{y, y, y, 255};
    }
}

// la fonction appelant le noyau pour calculer un niveau de gris
void niveauDeGrisGPU(BMP *h_imgRGB, BMP *h_imgGris,
                     float *elapsed_ms) {
    // declaration de variables
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, end;
    pixel *d_imgRGB, *d_imgGris;
    int width = get_width(h_imgRGB);
    int height = get_height(h_imgRGB);
    int pixels = width * height;
    int bytes = sizeof(pixel) * pixels;

    // Partie 1
    // Création des evenements
    err = cudaEventCreate(&start);
    EXPECT(err, "Couln't create start event");

    err = cudaEventCreate(&end);
    EXPECT(err, "Couln't create end event");

    // Allocate device memory for imgRGB and imgGris
    err = cudaMalloc((void **)&d_imgRGB, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgRGB");

    err = cudaMalloc((void **)&d_imgGris, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgGris");

    // copy imgRGB to device memory
    err = cudaMemcpy(d_imgRGB, (pixel *)h_imgRGB->pixels, bytes,
                     cudaMemcpyHostToDevice);
    EXPECT(err, "Could not copy h_imgRGB to d_imgRGB");

    // Partie 2
    // Kernel launch code
    // Dimensions de l'exécution
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    dim3 dimThreads(threadsPerBlock, threadsPerBlock, 1);

    dim3 dimBlocks(1, 1, 1);
    dimBlocks.x = (width + threadsPerBlock - 1) / threadsPerBlock;
    dimBlocks.y = (height + threadsPerBlock - 1) / threadsPerBlock;

    printf("niveauDeGrisKernel kernel launch with (%u, %u) "
           "blocks "
           "of (%u, %u) "
           "threads\n",
           dimBlocks.x, dimBlocks.y, dimThreads.x, dimThreads.y);

    // Exécution du kernel
    err = cudaEventRecord(start);
    EXPECT(err, "Couldn't not record start event");

    niveauDeGrisKernel<<<dimBlocks, dimThreads>>>(d_imgGris, d_imgRGB,
                                                  width, height);
    err = cudaGetLastError();
    EXPECT(err, "Error while executing kernel on GPU")

    err = cudaEventRecord(end);
    EXPECT(err, "Couln't not record end event");

    // Récupération tps d'exécution
    err = cudaEventSynchronize(end);
    EXPECT(err, "Error while syncronizing end event");

    err = cudaEventElapsedTime(elapsed_ms, start, end);
    EXPECT(err, "Error while measuring elapsed time");

    // Partie 3
    // copy imgGris from the device memory
    err = cudaMemcpy(h_imgGris->pixels, d_imgGris, bytes,
                     cudaMemcpyDeviceToHost);
    EXPECT(err, "Could not copy d_imgGRIS to h_imgGRIS");

    // Free device vectors
    err = cudaFree(d_imgRGB);
    EXPECT(err, "Could not free d_imgRGB");

    err = cudaFree(d_imgGris);
    EXPECT(err, "Could not free d_imgGris");
}

__global__ void BlurKernel(pixel *resImage, pixel *rgbImage,
                           int width, int height) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    // Accumulateurs
    int red = 0, green = 0, blue = 0;
    int pixels = 0;

    int blurRow, blurCol;
    if (row < height && col < width) {
        // Compute blur value
        for (blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE;
                 blurCol++) {
                int currRow = row + blurRow;
                int currCol = col + blurCol;
                if (currRow >= 0 && currRow < height &&
                    currCol >= 0 && currCol < width) {
                    pixel pix = rgbImage[width * currRow + currCol];
                    red += pix.red;
                    green += pix.green;
                    blue += pix.blue;
                    pixels++;
                }
            }
        }

        // Normalize & assign blur value
        resImage[width * row + col] =
            pixel{(unsigned char)(red / pixels),
                  (unsigned char)(green / pixels),
                  (unsigned char)(blue / pixels), 255};
    }
}

// la fonction appelant le noyau pour calculer le flou
void blurGPU(BMP *h_imgRGB, BMP *h_imgBlur, float *elapsed_ms) {
    // declaration de variables
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, end;

    pixel *d_imgRGB, *d_imgBlur;
    int width = get_width(h_imgRGB);
    int height = get_height(h_imgRGB);
    size_t pixels = width * height;
    size_t bytes = sizeof(pixel) * pixels;

    // Partie 1
    // Création des evenements
    err = cudaEventCreate(&start);
    EXPECT(err, "Couln't create start event");

    err = cudaEventCreate(&end);
    EXPECT(err, "Couln't create end event");

    // Allocate device memory for imgRGB and imgGris
    err = cudaMalloc((void **)&d_imgRGB, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgRGB");

    err = cudaMalloc((void **)&d_imgBlur, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgBlur");

    // copy imgRGB to device memory
    err = cudaMemcpy((void *)d_imgRGB, (void *)h_imgRGB->pixels,
                     bytes, cudaMemcpyHostToDevice);
    EXPECT(err, "Could not copy h_imgRGB to d_imgRGB");

    // copy imgRGB to device memory
    // Kernel launch code
    // Dimensions de l'exécution
    int threadsPerBlock = DEFAULT_THREADS_PER_BLOCK;
    dim3 dimThreads(threadsPerBlock, threadsPerBlock, 1);

    dim3 dimBlocks(1, 1, 1);
    dimBlocks.x = (width + threadsPerBlock - 1) / threadsPerBlock;
    dimBlocks.y = (height + threadsPerBlock - 1) / threadsPerBlock;

    printf("blurGPUKernel kernel launch with (%u, %u) "
           "blocks "
           "of (%u, %u) "
           "threads\n",
           dimBlocks.x, dimBlocks.y, dimThreads.x, dimThreads.y);

    // Exécution du kernel
    err = cudaEventRecord(start);
    EXPECT(err, "Couldn't not record start event");

    BlurKernel<<<dimBlocks, dimThreads>>>(d_imgBlur, d_imgRGB, width,
                                          height);
    err = cudaGetLastError();
    EXPECT(err, "Error while executing kernel on GPU")

    err = cudaEventRecord(end);
    EXPECT(err, "Couln't not record end event");

    // Récupération tps d'exécution
    err = cudaEventSynchronize(end);
    EXPECT(err, "Error while syncronizing end event");

    err = cudaEventElapsedTime(elapsed_ms, start, end);
    EXPECT(err, "Error while measuring elapsed time");

    // Partie 3
    // copy imgBlur from the device memory
    err = cudaMemcpy((void *)h_imgBlur->pixels, (void *)d_imgBlur,
                     bytes, cudaMemcpyDeviceToHost);
    EXPECT(err, "Could not copy d_imgBlur to h_imgBlur");

    // Free device vectors
    err = cudaFree(d_imgRGB);
    EXPECT(err, "Could not free d_imgRGB");

    err = cudaFree(d_imgBlur);
    EXPECT(err, "Could not free d_imgBlur");
}

__global__ void BlurKernelShared(pixel *resImage, pixel *rgbImage,
                                 int width, int height) {
    __shared__ pixel
        tile[TILE_SIZE + 2 * BLUR_SIZE][TILE_SIZE + 2 * BLUR_SIZE];
    int bx = blockDim.x * blockIdx.x;
    int by = blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = bx + tx;
    int row = by + ty;
    int offset = width * row + col;

    // Chargement de la tuile
    int tileRow, tileCol;
    int numTiles = (2 * (TILE_SIZE + BLUR_SIZE) - 1) / TILE_SIZE;
    int tileSize = TILE_SIZE + 2 * BLUR_SIZE;
    for (tileRow = 0; tileRow < numTiles; tileRow++) {
        for (tileCol = 0; tileCol < numTiles; tileCol++) {
            int tileX = tx + tileCol * TILE_SIZE;
            int tileY = ty + tileRow * TILE_SIZE;
            if (tileX < tileSize && tileY < tileSize) {
                int currCol =
                    CLAMP(bx + tileX - BLUR_SIZE, 0, width - 1);
                int currRow =
                    CLAMP(by + tileY - BLUR_SIZE, 0, height - 1);
                tile[tileY][tileX] =
                    rgbImage[width * currRow + currCol];
            }
        }
    }
    __syncthreads();

    // Accumulateurs
    int red = 0, green = 0, blue = 0;
    int pixels = 0;

    int blurRow, blurCol;
    if (row < height && col < width) {
        // Compute blur value
        for (blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; blurRow++) {
            for (blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE;
                 blurCol++) {
                int currCol = tx + blurCol + BLUR_SIZE;
                int currRow = ty + blurRow + BLUR_SIZE;
                if (currRow >= 0 && currRow < tileSize &&
                    currCol >= 0 && currCol < tileSize) {
                    pixel pix = tile[currRow][currCol];
                    red += pix.red;
                    green += pix.green;
                    blue += pix.blue;
                    pixels++;
                }
            }
        }

        // Normalize & assign blur value
        resImage[offset] = pixel{(unsigned char)(red / pixels),
                                 (unsigned char)(green / pixels),
                                 (unsigned char)(blue / pixels), 255};
    }
}

void blurGPU_Shared(BMP *h_imgRGB, BMP *h_imgBlur,
                    float *elapsed_ms) {
    // declaration de variables
    cudaError_t err = cudaSuccess;
    cudaEvent_t start, end;

    pixel *d_imgRGB, *d_imgBlur;
    int width = get_width(h_imgRGB);
    int height = get_height(h_imgRGB);
    size_t pixels = width * height;
    size_t bytes = sizeof(pixel) * pixels;

    // Partie 1
    // Création des evenements
    err = cudaEventCreate(&start);
    EXPECT(err, "Couln't create start event");

    err = cudaEventCreate(&end);
    EXPECT(err, "Couln't create end event");

    // Allocate device memory for imgRGB and imgGris
    err = cudaMalloc((void **)&d_imgRGB, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgRGB");

    err = cudaMalloc((void **)&d_imgBlur, bytes);
    EXPECT(err, "Could not allocate device memory for d_imgBlur");

    // copy imgRGB to device memory
    err = cudaMemcpy((void *)d_imgRGB, (void *)h_imgRGB->pixels,
                     bytes, cudaMemcpyHostToDevice);
    EXPECT(err, "Could not copy h_imgRGB to d_imgRGB");

    // copy imgRGB to device memory
    // Kernel launch code
    // Dimensions de l'exécution
    int threadsPerBlock = TILE_SIZE;
    dim3 dimThreads(threadsPerBlock, threadsPerBlock, 1);

    dim3 dimBlocks(1, 1, 1);
    dimBlocks.x = (width + threadsPerBlock - 1) / threadsPerBlock;
    dimBlocks.y = (height + threadsPerBlock - 1) / threadsPerBlock;

    printf("blurGPUKernel_Shared kernel launch with (%u, %u) "
           "blocks "
           "of (%u, %u) "
           "threads\n",
           dimBlocks.x, dimBlocks.y, dimThreads.x, dimThreads.y);

    // Exécution du kernel
    err = cudaEventRecord(start);
    EXPECT(err, "Couldn't not record start event");

    BlurKernelShared<<<dimBlocks, dimThreads>>>(d_imgBlur, d_imgRGB,
                                                width, height);
    err = cudaGetLastError();
    EXPECT(err, "Error while executing kernel on GPU")

    err = cudaEventRecord(end);
    EXPECT(err, "Couln't not record end event");

    // Récupération tps d'exécution
    err = cudaEventSynchronize(end);
    EXPECT(err, "Error while syncronizing end event");

    err = cudaEventElapsedTime(elapsed_ms, start, end);
    EXPECT(err, "Error while measuring elapsed time");

    // Partie 3
    // copy imgBlur from the device memory
    err = cudaMemcpy((void *)h_imgBlur->pixels, (void *)d_imgBlur,
                     bytes, cudaMemcpyDeviceToHost);
    EXPECT(err, "Could not copy d_imgBlur to h_imgBlur");

    // Free device vectors
    err = cudaFree(d_imgRGB);
    EXPECT(err, "Could not free d_imgRGB");

    err = cudaFree(d_imgBlur);
    EXPECT(err, "Could not free d_imgBlur");
}
