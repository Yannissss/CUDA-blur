### Flou gaussien CUDA avec mémoire partagée

#### Chargement des modules

Rentrez les commandes suivant sur ROMEO afin de chargement les modules nécéssaires à la compilation et au fonctionnement de l'application
```bash
source env.sh
```
ou
```bash
module load gcc/10.2
module load llvm/14
module load cuda/11.4
```

### Compilation

Compilez avec make
```bash
make
```

### Exécution

Lancer le programme avec ```srun -N 1 --gres=gpu:1 ./cuda-flou```:

Exemple de sortie:
```
Ouverture de l'image ./images/monImage.bmp
Taille de l'image : 1280 x 720
Echauffement (~ 1s)...

@ niveauDeGris        : 3686400 R/Ws, 4608000 OPs 
niveauDeGrisKernel kernel launch with (80, 45) blocks of (16, 16) threads
CPU Time              : 1.97ms
GPU Time              : 40.10µs
Threads/block         : 256 
Total Memory          : 3.69 MiB 
Effective bandwidth   : 91.94 GiB/s 
Effective throughput  : 114.92 GOP/s 
=> Ecriture dans ./images/monImageGris.bmp

@ blurGPU_naif        : 95846400 R/Ws, 70041600 OPs 
blurGPUKernel kernel launch with (80, 45) blocks of (16, 16) threads
CPU Time              : 2.44ms
GPU Time              : 633.79µs
Threads/block         : 256 
Total Memory          : 3.69 MiB 
Effective bandwidth   : 151.23 GiB/s 
Effective throughput  : 110.51 GOP/s 
=> Ecriture dans ./images/monImageBlur.bmp

@ blurGPU_shared      : 10177920 R/Ws, 70041600 OPs 
blurGPUKernel_Shared kernel launch with (40, 23) blocks of (32, 32) threads
CPU Time              : 2.16ms
GPU Time              : 224.90µs
Threads/block         : 1024 
Total Memory          : 3.69 MiB 
Effective bandwidth   : 45.26 GiB/s 
Effective throughput  : 311.44 GOP/s 
=> Ecriture dans ./images/monImageShared.bmp
```