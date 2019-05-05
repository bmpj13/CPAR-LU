#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <assert.h>
#include <CL/cl.hpp>
#include <fstream>

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

void print1DMatrix(double* m, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("| %6.2f |", m[i*size + j]);
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

double* get1DArray(double** copy_matrix, int size) {
    double *m;

    m = (double *) malloc(size * size * sizeof(double));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            m[i*size + j] = copy_matrix[i][j];
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

    begin = clock();
    int k = 0;
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
 *  ----- ----------
 * | A00 |    A01   |
 *  ----- ----------
 * |     |          |
 * | A10 |    A11   |
 * |     |          |
 *  ----- ----------
 * 
 * */
double decomposeSequentialBlock(double **m, int size, int block) {
    clock_t begin, end;

    begin = clock();
    for (int k0 = 0; k0 < size; k0 += block) {
        int limit = MIN(k0+block, size);
  
        for (int k = k0; k < limit && m[k][k] != 0; k++) {
            // A00 + A10
            for (int i = k+1; i < size; i++) {
                m[i][k] /= m[k][k];
                for (int j = k+1; j < limit; j++) {
                    m[i][j] -= m[i][k] * m[k][j];
                }
            }

            // A01
            for (int i = k+1; i < limit; i++) {
                for (int j = limit; j < size; j++) {
                    m[i][j] -= m[i][k] * m[k][j];
                }
            }
        }

        // A11
        for (int i = limit; i < size; i++) {
            for (int k = k0; k < limit; k++) {
                for (int j = limit; j < size; j++) {
                    m[i][j] -= m[i][k] * m[k][j];
                }
            }
        }
    }
    end = clock();

    return (double) (end - begin) / CLOCKS_PER_SEC;
}

double decomposeParallelMP(double **m, int size) {
    double begin, end;
    int k;

    omp_set_num_threads(10); // should use what?
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

    /*
    // Single-dimension array
    std::vector<int> vec(1024, 5);
    cl::Buffer inBuf(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, sizeof(int) * vec.size(), vec.data());
    cl::Buffer outBuf(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(int) * vec.size());
    cl::Kernel kernel(program, "ProcessArray");

    kernel.setArg(0, inBuf);
    kernel.setArg(1, outBuf);
    cl::CommandQueue queue(context, device);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec.size()));
    queue.enqueueReadBuffer(outBuf, CL_FALSE, 0, sizeof(int) * vec.size(), vec.data());
    cl::finish();
    */
    
    /*
    // Hello World!
    char buf[16];
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, sizeof(buf));
    cl::Kernel kernel(program, "HelloWorld");

    kernel.setArg(0, buffer);

    cl::CommandQueue queue(context, device);
    queue.enqueueTask(kernel);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(buf), buf);

    printf("RESULT: %s", buf);
    */
}

int main(int argc, char **argv) {
    srand(time(NULL));

    double **m1, **m2, **m3, *m4, elapsed;
    int size = 1000;
    int block = size / 10;

    
    m1 = generateMatrix(size);
    m2 = copyMatrix(m1, size);
    m3 = copyMatrix(m1, size);
    m4 = get1DArray(m1, size);

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
    //print1DMatrix(m4, size);
    
    freeMatrix(m1, size);
    freeMatrix(m2, size);
    freeMatrix(m3, size);
    free(m4);

    return 0;
}