#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void printMatrix(double **m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("| %6.2f |", m[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixL(double **m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            printf("| %6.2f |", m[i][j]);
        }
        
        printf("| %6.2f |", 1.0);

        for (int j = i+1; j < size; j++) {
            printf("| %6.2f |", 0.0);
        }

        printf("\n");
    }
    printf("\n");
}

void printMatrixU(double **m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            printf("| %6.2f |", 0.0);
        }

        for (int j = i; j < size; j++) {
            printf("| %6.2f |", m[i][j]);
        }

        printf("\n");
    }
    printf("\n");
}

double ** generateMatrix(int size) {
    double **m;

    m = (double **) malloc(size * sizeof(*m));

    for (int i = 0; i < size; i++) {
        m[i] = (double *) malloc(size * sizeof(*m[i]));

        for (int j = 0; j < size; j++) {
            m[i][j] = i*2 + j + 1;  // no specific reason
        }
    }

    return m;
}

void freeMatrix(double **m, int size) {
    for(int i = 0; i < size; i++)
        free(m[i]);
    free(m);
}

double decomposeSequential(double **m, int size) {
    clock_t begin, end;
    int k = 0;

    begin = clock();
    while (k < size && m[k][k] != 0) {
        for (int i = k+1; i < size; i++) {
            m[i][k] /= m[k][k];

            for (int j = k+1; j < size; j++) {
                m[i][j] -= m[i][k] * m[k][j];
            }
        }
        k++;
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

/**
 * ?????????????????????????????????????????????????????????????
 * */
double decomposeSequentialBlock(double **m, int size, int block) {
    clock_t begin, end;
    int k = 0;

    begin = clock();
    while (k < size && m[k][k] != 0) {
        for (int i = k+1; i < size; i++) {  // Colocar dentro de blocos?????
            m[i][k] /= m[k][k];
        }

        for (int i0 = k+1; i0 < size; i0 += block) {
            int limitI = MIN(i0+block, size);

            for (int j0 = k+1; j0 < size; j0 += block) {
                int limitJ = MIN(j0+block, size);

                for (int i = i0; i < limitI; i++) {
                    for (int j = j0; j < limitJ; j++) {
                        m[i][j] -= m[i][k] * m[k][j];
                    }
                }
            }
        }
        k++;
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

double decomposeParallel(double **m, int size) {
    clock_t begin, end;
    int k;

    omp_set_num_threads(4); // should use what?
    begin = clock();
    #pragma omp parallel private(k)
    {
        k = 0;
        while (m[k][k] != 0 && k < size) {
            #pragma omp for
            for (int i = k+1; i < size; i++) {
                m[i][k] /= m[k][k];
            }
            
            #pragma omp for
            for (int i = k+1; i < size; i++) {
                for (int j = k+1; j < size; j++) {
                    m[i][j] -= m[i][k] * m[k][j];
                }
            }

            k++;
        }
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    double **m, elapsed;
    int size = 50000;
    int block = 300;

    m = generateMatrix(size);
    elapsed = decomposeSequential(m, size);
    printf("\nElapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m, size);
    freeMatrix(m, size);
    
    m = generateMatrix(size);
    elapsed = decomposeSequentialBlock(m, size, block);
    printf("\nElapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m, size);
    freeMatrix(m, size);

    m = generateMatrix(size);
    elapsed = decomposeParallel(m, size);
    printf("Elapsed time: %6.3f seconds\n\n", elapsed);
    //printMatrix(m, size);
    freeMatrix(m, size);

    return 0;
}