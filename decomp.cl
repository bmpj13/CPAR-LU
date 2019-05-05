__kernel void HelloWorld(__global char* data)
{
    data[0] = 'H';
    data[1] = 'e';
    data[2] = 'l';
    data[3] = 'l';
    data[4] = 'o';
    data[5] = ' ';
    data[6] = 'W';
    data[7] = 'o';
    data[8] = 'r';
    data[9] = 'l';
    data[10] = 'd';
    data[11] = '!';
    data[12] = '\n';
}

__kernel void ProcessArray(__global int* data, __global int* outData)
{
    outData[get_global_id(0)] = data[get_global_id(0)] * 2;
}

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