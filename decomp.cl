__kernel void ProcessColumn(__global double* data, int size, int k) {
    int id = get_global_id(0);
    int row = k + id + 1;

    data[row*size + k] /= data[k*size + k];
}

__kernel void ProcessSubmatrix(__global double* data, int size, int k) {
    int idRow = get_global_id(0);
    int idCol = get_global_id(1);
    int row = k + idRow + 1;
    int col = k + idCol + 1;

    data[row*size + col] -= data[row*size + k] * data[k*size + col];
}

__kernel void ProcessColumnBlocks(__global double* data, int size, int k, int blockSize) {
    int id = get_global_id(0);
    int start = id * blockSize;

    printf("id: %d; blockSize: %d; start: %d\n", id, blockSize, start);

    for (int i = start; i < start + blockSize; i++) {
        int row = k + i + 1; // <----- TODO PRECISO TROCAR ISTO!!
        printf("size: %d; i: %d, j: %d\n", size, row*size + k, k*size + k);
        data[row*size + k] /= data[k*size + k];
    }
}

__kernel void ProcessSubmatrixBlocks(__global double* data, int size, int k, int blockSize) {
    int idRow = get_global_id(0);
    int idCol = get_global_id(1);
    int startI = idRow * blockSize;
    int startJ = idCol * blockSize;

    for (int i = startI; i < startI + blockSize; i++) {
        for (int j = startJ; j < startJ + blockSize; j++) {
            int row = k + i + 1;
            int col = k + j + 1;
            data[row*size + col] -= data[row*size + k] * data[k*size + col];
        }
    }
}