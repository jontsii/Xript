#include <windows.h>

#define EXPORT __declspec(dllexport)

typedef void (*fptr)(void*);
typedef struct {fptr func;
                void* args;} WINThreadArg; // args is a a pointer t o a struct that can be used as an argument
                
DWORD WINAPI WINthread(LPVOID lpParam) { //func will be used INSANELY carefully
    WINThreadArg* data = (WINThreadArg*)lpParam;
    data[0].func(data[0].args);
    return 0;
}
EXPORT HANDLE createWINThread(void* args) { //create thread without ID
    HANDLE thread = CreateThread(NULL, 0, WINthread, (LPVOID)&args, CREATE_SUSPENDED, NULL);
    return thread;
}
EXPORT void startWINThread(HANDLE thread) {
    ResumeThread(thread);
}
EXPORT void joinWINThread(HANDLE thread) {
    WaitForSingleObject(thread, INFINITE);
}
EXPORT void killWINThread(HANDLE thread) {
    TerminateThread(thread, 1);
}