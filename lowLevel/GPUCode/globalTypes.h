typedef struct {
    float x, y, z;
} V3;
//float* data is just a 1 dimensional array. Each item of it is accessed by the rows and columns, but its turned into an index like this: rowNumber * height + columnNumber
typedef struct {
    int xLen;
    int yLen;
    int size;
    float* data;
} M2;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int size;
    float* data;
} M3;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int aLen;
    int size;
    float* data;
} M4;

typedef struct {
    int xLen;
    int yLen;
    int zLen;
    int aLen;
    int bLen;
    int size;
    float* data;
} M5;