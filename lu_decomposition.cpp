#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <CL/cl.hpp>
#include <fstream>
#include <math.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))

void printMatrix(double *m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("| %6.2f |", m[i*size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrixL(double *m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            printf("| %6.2f |", m[i*size + j]);
        }
        
        printf("| %6.2f |", 1.0);

        for (int j = i+1; j < size; j++) {
            printf("| %6.2f |", 0.0);
        }

        printf("\n");
    }
    printf("\n");
}

void printMatrixU(double *m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            printf("| %6.2f |", 0.0);
        }

        for (int j = i; j < size; j++) {
            printf("| %6.2f |", m[i*size + j]);
        }

        printf("\n");
    }
    printf("\n");
}

double* generateMatrix(int size) {
    double *m;

    m = (double *) malloc(size * size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            m[i*size + j] = (double) ((rand() % 20) + 1);
        }
    }

    return m;
}

double* copyMatrix(double* copy_matrix, int size) {
    double *m;

    m = (double *) malloc(size * size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            m[i*size + j] = copy_matrix[i*size + j];
        }
    }

    return m;
}

bool equalMatrixes(double *m1, double *m2, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (abs(m1[i*size + j] - m2[i*size + j]) > 5.0) {
                // printf("NOT EQUAL AT (i,j) = (%d,%d); m1: %f, m2: %f\n", i, j, m1[i*size + j], m2[i*size + j]);
                return false;
            }
        }
    }

    return true;
}

double decomposeSequential(double *m, int size) {
    clock_t begin, end;

    begin = clock();
    int k = 0;
    while (k < size && m[k*size + k] != 0) {
        for (int i = k+1; i < size; i++) {
            m[i*size + k] /= m[k*size + k];

            for (int j = k+1; j < size; j++) {
                m[i*size + j] -= m[i*size + k] * m[k*size + j];
            }
        }
        k++;
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

/**
 *  ----- ----------
 * | A00 |    A01   |
 *  ----- ----------
 * |     |          |
 * | A10 |    A11   |
 * |     |          |
 *  ----- ----------
 * */
double decomposeSequentialBlock(double *m, int size, int block) {
    clock_t begin, end;

    begin = clock();
    for (int k0 = 0; k0 < size; k0 += block) {
        int limit = MIN(k0+block, size);
  
        for (int k = k0; k < limit && m[k*size + k] != 0; k++) {
            // A00 (green) + A10 (yellow)
            for (int i = k+1; i < size; i++) {
                m[i*size + k] /= m[k*size + k];
                for (int j = k+1; j < limit; j++) {
                    m[i*size + j] -= m[i*size + k] * m[k*size + j];
                }
            }

            // A01 (red)
            for (int i = k+1; i < limit; i++) {
                for (int j = limit; j < size; j++) {
                    m[i*size + j] -= m[i*size + k] * m[k*size + j];
                }
            }
        }

        // A11 (blue)
        for (int i = limit; i < size; i++) {
            for (int k = k0; k < limit; k++) {
                for (int j = limit; j < size; j++) {
                    m[i*size + j] -= m[i*size + k] * m[k*size + j];
                }
            }
        }
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

double decomposeParallelMP(double *m, int size) {
    double begin, end;
    int k;

    omp_set_num_threads(1);
    begin = omp_get_wtime();
    #pragma omp parallel private(k)
    {
        k = 0;
        while (k < size && m[k*size + k] != 0) {

            #pragma omp for
            for (int i = k+1; i < size; i++) {
                m[i*size + k] /= m[k*size + k];
            }
            
            #pragma omp for
            for (int i = k+1; i < size; i++) {
                for (int j = k+1; j < size; j++) {
                    m[i*size + j] -= m[i*size + k] * m[k*size + j];
                }
            }

            k++;
        }
    }
    end = omp_get_wtime();

    return (double) (end - begin);
}

double decomposeParallelCL(double *m, int size) {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    clock_t begin, end;

    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);
    cl::Platform platform = platforms.front();
    
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size() > 0);
    cl::Device device = devices.front();

    std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
    std::string version = device.getInfo<CL_DEVICE_VERSION>();
    printf("DEVICE: %s %s\n", vendor.c_str(), version.c_str());

    std::ifstream decompFile("decomp.cl");
    std::string src((std::istreambuf_iterator<char>(decompFile)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length()+1));
    cl::Context context(devices);
    cl::Program program(context, sources);

    program.build();
    
    cl::CommandQueue queue(context, device);
    cl::Buffer matrix(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * size * sizeof(double), m);

    cl::Kernel kernelColumns(program, "ProcessColumn");
    kernelColumns.setArg(0, matrix);
    kernelColumns.setArg(1, size);

    cl::Kernel kernelSubmatrix(program, "ProcessSubmatrix");
    kernelSubmatrix.setArg(0, matrix);
    kernelSubmatrix.setArg(1, size);

    begin = clock();
    int k = 0;
    while (k < size && m[k*size + k] != 0) {
        kernelColumns.setArg(2, k);
        kernelSubmatrix.setArg(2, k);
        queue.enqueueNDRangeKernel(kernelColumns, cl::NullRange, cl::NDRange(size-1-k));
        queue.enqueueNDRangeKernel(kernelSubmatrix, cl::NullRange, cl::NDRange(size-1-k, size-1-k));
        k++;
    }
    cl::finish();
    end = clock();
    
    queue.enqueueReadBuffer(matrix, CL_TRUE, 0, size * size * sizeof(double), m);
    return (double) (end - begin) / CLOCKS_PER_SEC;
}

double decomposeParallelCLBlocks(double *m, int size, int cores) {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    clock_t begin, end;

    cl::Platform::get(&platforms);
    assert(platforms.size() > 0);
    cl::Platform platform = platforms.front();
    
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    assert(devices.size() > 0);
    cl::Device device = devices.front();

    std::string vendor = device.getInfo<CL_DEVICE_VENDOR>();
    std::string version = device.getInfo<CL_DEVICE_VERSION>();
    int numCores = MIN(cores, device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
    printf("DEVICE: %s %s; Number of cores: %d\n", vendor.c_str(), version.c_str(), numCores);

    std::ifstream decompFile("decomp.cl");
    std::string src((std::istreambuf_iterator<char>(decompFile)), std::istreambuf_iterator<char>());

    cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length()+1));
    cl::Context context(devices);
    cl::Program program(context, sources);

    program.build();
    
    cl::CommandQueue queue(context, device);
    cl::Buffer matrix(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR, size * size * sizeof(double), m);

    cl::Kernel kernelColumns(program, "ProcessColumnBlocks");
    kernelColumns.setArg(0, matrix);
    kernelColumns.setArg(1, size);

    cl::Kernel kernelSubmatrix(program, "ProcessSubmatrixBlocks");
    kernelSubmatrix.setArg(0, matrix);
    kernelSubmatrix.setArg(1, size);

    begin = clock();
    int k = 0;
    while (k < size && m[k*size + k] != 0) {
        int blockSize = (int) ceil((double) (size-1-k)/numCores);
        int range = MIN(size-1-k, numCores);
        kernelColumns.setArg(2, k);
        kernelSubmatrix.setArg(2, k);
        kernelColumns.setArg(3, blockSize);
        kernelSubmatrix.setArg(3, blockSize);
        queue.enqueueNDRangeKernel(kernelColumns, cl::NullRange, cl::NDRange(range));
        queue.enqueueNDRangeKernel(kernelSubmatrix, cl::NullRange, cl::NDRange(range, range));
        k++;
    }
    cl::finish();
    end = clock();
    
    queue.enqueueReadBuffer(matrix, CL_TRUE, 0, size * size * sizeof(double), m);
    return (double) (end - begin) / CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    srand(time(NULL));

    double *m, *m1, *m2, *m3, *m4, *m5, elapsed;
    int size = 2828;
    int block = size / 10;
    int cores = 3;
    
    m = generateMatrix(size);
    m1 = copyMatrix(m, size);
    m2 = copyMatrix(m, size);
    m3 = copyMatrix(m, size);
    m4 = copyMatrix(m, size);
    m5 = copyMatrix(m, size);

    elapsed = decomposeSequential(m1, size);
    printf("\nElapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m1, size);
    
    elapsed = decomposeSequentialBlock(m2, size, block);
    printf("Elapsed time: %6.3f seconds\n", elapsed);
    //printMatrix(m2, size);

    elapsed = decomposeParallelMP(m3, size);
    printf("Elapsed time: %6.3f seconds\n\n", elapsed);
    //printMatrix(m3, size);
    
    elapsed = decomposeParallelCL(m4, size);
    printf("Elapsed time: %6.3f seconds\n\n", elapsed);
    //printMatrix(m4, size);

    elapsed = decomposeParallelCLBlocks(m5, size, cores);
    printf("Elapsed time: %6.3f seconds\n\n", elapsed);
    //printMatrix(m5, size);

    if (
        !equalMatrixes(m, m1, size) &&
        equalMatrixes(m1, m2, size) &&
        equalMatrixes(m1, m3, size) &&
        equalMatrixes(m1, m4, size) &&
        equalMatrixes(m1, m5, size)
    ) {
        printf("CORRECT RESULT\n");
    }
    
    free(m);
    free(m1);
    free(m2);
    free(m3);
    free(m4);
    free(m5);

    return 0;
}
