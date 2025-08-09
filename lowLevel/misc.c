#include <stdio.h>
#include <windows.h>

typedef char* string;
//io stuff:

void writeLn(string msg) {
    printf(msg);
}
string readLn() {
    string input;
    fgets(input, sizeof(input), stdin);
    return input;
}