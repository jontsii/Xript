#include <stdio.h>
#include <stdlib.h>

typedef char* string;

string readFile(string path) {
    FILE* f = fopen(path, "r");
    string fileData;
    fscanf(f, "%s", &fileData);

    fclose(f);
    return fileData;
}
void writeFile(string path, string data) {
    FILE* f = fopen(path, "w");
    fprintf(f, data);

    fclose(f);
}
void deleteFile(string path) {
    remove(path);
}
void moveFile(string srcPath, string destPath) {
    string data = readFile(srcPath);
    writeFile(destPath, data);
    remove(srcPath);
}
void copyFile(string srcPath, string destPath) {
    string data = readFile(srcPath);
    writeFile(destPath, data);
}