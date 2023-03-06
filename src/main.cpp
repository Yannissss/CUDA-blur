#ifdef __clang__
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include "cbmp.h"
#include "imageGPU.h"
#include "stopwatch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Mesure la performance d'un filtre cuda appliqué à notre image
void bench(BMP *In, BMP *Out, const char *name, int threadsPerBlock,
           int bytes, int readsWrites, int ops,
           void (*filter)(BMP *, BMP *, float *)) {
    Stopwatch sw;
    float elapsed_ms;

    printf("@ %-20s: %d R/Ws, %d OPs \n", name, readsWrites, ops);

    // Calcul
    sw.start();
    filter(In, Out, &elapsed_ms);
    sw.stop();

    // Affichage des statistiques d'exécution

    printf("%-22s: ", "CPU Time");
    sw.print_human_readable();
    printf("\n");
    printf("%-22s: ", "GPU Time");
    Stopwatch::print_elapsed_ms(elapsed_ms);
    printf("\n");

    printf("%-22s: %d \n", "Threads/block", threadsPerBlock);
    printf("%-22s: %.2f MiB \n", "Total Memory", bytes / 1e6);
    printf("%-22s: %.2f GiB/s \n", "Effective bandwidth",
           readsWrites / elapsed_ms / 1e6);
    printf("%-22s: %.2f GOP/s \n", "Effective throughput",
           ops / elapsed_ms / 1e6);
}

int main(int argc, char **argv) {
    // Pour les images
    BMP *In, *Out;
    int width, height;
    int bytes;
    char nomIn[100], nomOut[100];

    // Récupération des paramètres
    // TODO

    // fonctionne avec le nom de fichier par defaut "monImage.bmp"
    strcpy(nomIn, "./images/monImage");
    strcat(nomIn, ".bmp");

    // Lecture/chargement de l'image
    In = bopen(nomIn);
    printf("Ouverture de l'image %s\n", nomIn);
    if (In == NULL) {
        printf("Le fichier %s n'exite pas\n", nomIn);
        exit(EXIT_FAILURE);
    }
    width = get_width(In);
    height = get_height(In);
    bytes = sizeof(pixel) * width * height;
    printf("Taille de l'image : %d x %d\n", width, height);

    // Application filtres

    // Echauffement
    {
        float _0;
        Out = b_deep_copy(In);
        printf("Echauffement (~ 1s)...\n");
        echauffement();
        bclose(Out);
    }
    // Niveau de gris
    {
        Out = b_deep_copy(In);
        bench(In, Out, "niveauDeGris",
              DEFAULT_THREADS_PER_BLOCK * DEFAULT_THREADS_PER_BLOCK,
              bytes, bytes, 5 * width * height, niveauDeGrisGPU);
        strcpy(nomOut, "./images/monImageGris.bmp");
        printf("=> Ecriture dans %s\n", nomOut);
        bwrite(Out, nomOut);
        bclose(Out);
    }
    // Blur naif
    {
        Out = b_deep_copy(In);
        bench(In, Out, "blurGPU_naif",
              DEFAULT_THREADS_PER_BLOCK * DEFAULT_THREADS_PER_BLOCK,
              bytes, (BLUR_SIZE * BLUR_SIZE + 1) * bytes,
              (BLUR_SIZE * BLUR_SIZE * 3 + 1) * width * height,
              blurGPU);
        strcpy(nomOut, "./images/monImageBlur.bmp");
        printf("=> Ecriture dans %s\n", nomOut);
        bwrite(Out, nomOut);
        bclose(Out);
    }
    // Blur shared
    {
        Out = b_deep_copy(In);
        bench(In, Out, "blurGPU_shared", TILE_SIZE * TILE_SIZE, bytes,
              ((width + TILE_SIZE - 1) / TILE_SIZE) *
                      ((height + TILE_SIZE - 1) / TILE_SIZE) *
                      (TILE_SIZE + 2 * BLUR_SIZE) *
                      (TILE_SIZE + 2 * BLUR_SIZE) * sizeof(pixel) +
                  bytes,
              (BLUR_SIZE * BLUR_SIZE * 3 + 1) * width * height,
              blurGPU_Shared);
        strcpy(nomOut, "./images/monImageShared.bmp");
        printf("=> Ecriture dans %s\n", nomOut);
        bwrite(Out, nomOut);
        bclose(Out);
    }

    // Clean-up
    bclose(In);

    return EXIT_SUCCESS;
}
