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
    int start = (id * blockSize) + k + 1;
    int end = size < start+blockSize ? size : start+blockSize;

    for (int i = start; i < end; i++) {
        data[i*size + k] /= data[k*size + k];
    }
}

__kernel void ProcessSubmatrixBlocks(__global double* data, int size, int k, int blockSize) {
    int idRow = get_global_id(0);
    int idCol = get_global_id(1);
    int startI = (idRow * blockSize) + k + 1;
    int startJ = (idCol * blockSize) + k + 1;
    int endI = size < startI+blockSize ? size : startI+blockSize;
    int endJ = size < startJ+blockSize ? size : startJ+blockSize;

    for (int i = startI; i < endI; i++) {
        for (int j = startJ; j < endJ; j++) {
            data[i*size + j] -= data[i*size + k] * data[k*size + j];
        }
    }
}