#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>

#include <unistd.h>

int main(int argc, char *argv[]) {
    char *soPath = argv[1];
    char *socketFile = "/data/data/com.termux/files/dlopensocketserver.sock";

    int sock = socket(AF_UNIX, SOCK_DGRAM, 0);
    sockaddr_un addr;
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, socketFile);
    connect(sock, (sockaddr*)&addr, sizeof(addr));
    printf("Sending .so with path %s to socket server\n", soPath);
    send(sock, soPath, strlen(soPath), 0);

    // Don't exit until get interrupt. Hack to make jupyter-client think kernel is still running.
    while (true) {
        sleep(1);
    }

    return 0;
}