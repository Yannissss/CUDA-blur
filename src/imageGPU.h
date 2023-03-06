#ifndef _IMAGEGPU_H
#define _IMAGEGPU_H

#include "cbmp.h"

#define BLUR_SIZE (5)
#define TILE_SIZE (32)
#define DEFAULT_THREADS_PER_BLOCK (16)

void echauffement();

void niveauDeGrisGPU(BMP *h_imgRGB, BMP *h_imgGris,
                     float *elapsed_ms);
void blurGPU(BMP *h_imgRGB, BMP *h_imgBlur, float *elapsed_ms);

void blurGPU_Shared(BMP *h_imgRGB, BMP *h_imgBlur, float *elapsed_ms);

#endif
