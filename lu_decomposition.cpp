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
            m[i][j] = (double) ((rand() % 20) + 1);
        }
    }

    return m;
}

double ** copyMatrix(double** copy_matrix, int size) {
    double **m;

    m = (double **) malloc(size * sizeof(*m));

    for (int i = 0; i < size; i++) {
        m[i] = (double *) malloc(size * sizeof(*m[i]));

        for (int j = 0; j < size; j++) {
            m[i][j] = copy_matrix[i][j];
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
        for (int i0 = k+1; i0 < size; i0+=block) {

            for (int i = i0; i < MIN(i0+block, size); i++) {
                m[i][k] /= m[k][k];
                for (int j = k+1; j < size; j++) {
                    m[i][j] -= m[i][k] * m[k][j];
                }
            }
            
        }
        k++;
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

double decomposeParallel(double **m, int size) {
    double begin, end;
    int k;

    omp_set_num_threads(4); // should use what?
    begin = omp_get_wtime();
    #pragma omp parallel private(k)
    {
        k = 0;
        while (k < size && m[k][k] != 0) {

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
    end = omp_get_wtime();

    return (double) (end - begin);
}

int main(int argc, char **argv) {
    srand(time(NULL));

    double **m1, **m2, **m3, elapsed;
    int size = 2000;
    int block = size / 10;

    m1 = generateMatrix(size);
    m2 = copyMatrix(m1, size);
    m3 = copyMatrix(m1, size);

    elapsed = decomposeSequential(m1, size);
    printf("\nElapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m1, size);
    
    elapsed = decomposeSequentialBlock(m2, size, block);
    printf("Elapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m2, size);

    elapsed = decomposeParallel(m3, size);
    printf("Elapsed time: %6.3f seconds\n\n", elapsed);
    //printMatrix(m3, size);

    freeMatrix(m1, size);
    freeMatrix(m2, size);
    freeMatrix(m3, size);

    return 0;
}