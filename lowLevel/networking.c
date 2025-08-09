#include <WinSock2.h>
#include <string.h>
#include <stdbool.h>
#pragma comment(lib, "ws3_32.lib")

#define EXPORT __declspec(dllexport)

//DANGER!!!! absolute shit code below ↓
typedef char* string;
typedef struct {WSADATA wsa;
                SOCKET sock, clientSock;
                struct sockaddr_in server, client;
                int addrLen;} EXPORT TCPReceiverData;

typedef struct {WSADATA wsa;
                SOCKET sock;
                struct sockaddr_in target;} EXPORT TCPSenderData;


//TCPServer
EXPORT TCPReceiverData TCPReceiverinit(int port) {
    TCPReceiverData TR;
    WSAStartup(MAKEWORD(2, 2), &TR.wsa);

    TR.sock = socket(AF_INET, SOCK_STREAM, 0);
    TR.addrLen = sizeof(INADDR_ANY);

    TR.server.sin_family = AF_INET;
    TR.server.sin_addr.s_addr = htonl(INADDR_ANY);
    TR.server.sin_port = htons(port);

    return TR;
}
EXPORT string TCPReceiverReceive(TCPReceiverData TR) {
    listen(TR.sock, SOMAXCONN);

    string buffer = malloc(sizeof(char) * 1024);
    TR.clientSock = accept(TR.sock, (struct sockaddr*)&TR.client, &TR.addrLen);

    int recvLen = recv(TR.clientSock, buffer, 1024, 0);
    buffer[recvLen] = '\0';;
    return buffer;
}
EXPORT void TCPReceiverCleanUp(TCPReceiverData TR) {
    closesocket(TR.clientSock);
    closesocket(TR.sock);
    WSACleanup();
}

//TCP client
EXPORT TCPSenderData TCPSenderInit() { 
    TCPSenderData TS;
    WSAStartup(MAKEWORD(2, 2), &TS.wsa);
    TS.sock = socket(AF_INET, SOCK_STREAM, 0);

    return TS;
}
EXPORT void TCPSendersend(TCPSenderData TS, string target, string msg, int port) {
    TS.target.sin_addr.s_addr = inet_addr(target);
    TS.target.sin_family = AF_INET;
    TS.target.sin_port = htons(port);

    connect(TS.sock, (struct sockaddr*)&TS.target, sizeof(TS.target));
    send(TS.sock, msg, strlen(msg), 0);
}
EXPORT void TCPSenderCleanUp(TCPSenderData WS) {
    closesocket(WS.sock);
    WSACleanup();
}

// UDP stuff

//UDP server
typedef struct {WSADATA wsa;
                SOCKET sock;
                struct sockaddr_in server, client;
                int clientLen;} EXPORT UDPReceiverData;

typedef struct {WSADATA wsa;
                SOCKET sock;
                struct sockaddr_in target;} EXPORT UDPSenderData;


EXPORT UDPReceiverData UDPReceiverInit(int port) {
    UDPReceiverData UR;
    WSAStartup(MAKEWORD(2, 2), &UR.wsa);
    UR.sock = socket(AF_INET, SOCK_DGRAM, 0);

    UR.server.sin_family = AF_INET;
    UR.server.sin_addr.s_addr = htonl(INADDR_ANY);
    UR.server.sin_port = htons(port);

    bind(UR.sock, (struct sockaddr*)&UR.server, sizeof(UR.server));
    return UR;
}
EXPORT string UDPReceiveiverReceive(UDPReceiverData UR) {
    string buffer = malloc(sizeof(char) * 1024);
    int clientSize = sizeof(UR.client);
    int recvLen = recvfrom(UR.sock, buffer, 1024, 0, (struct sockaddr*)&UR.client, &clientSize);

    buffer[recvLen] = '\0';
    return buffer;
}
EXPORT void UDPReceiverCleanUp(UDPReceiverData UR) {
    closesocket(UR.sock);
    WSACleanup();
}

//UDP client
EXPORT UDPSenderData UDPSenderInit() {
    UDPSenderData US;
    WSAStartup(MAKEWORD(2, 2), &US.wsa);
    US.sock = socket(AF_INET, SOCK_DGRAM, 0);
    return US;
}
EXPORT void UDPSenderSetTarget(UDPSenderData US, string target, int port) {
    US.target.sin_family = AF_INET;
    US.target.sin_addr.s_addr = inet_addr(target);
    US.target.sin_port = htons(port);
}
EXPORT void UDPSenderSend(UDPSenderData US, string msg) {
    sendto(US.sock, msg, strlen(msg), 0, (struct sockaddr*)&US.target, sizeof(US.target));
}
EXPORT void UDPSenderCleanUp(UDPSenderData US) {
    closesocket(US.sock);
    WSACleanup();
}