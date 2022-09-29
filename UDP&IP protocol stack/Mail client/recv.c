#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>

#define MAX_SIZE 65535

char buf[MAX_SIZE+1];

// deal with resp
void recv_resp_info(int resp, char temp[]){
    if(resp == -1){
        perror("recv");
        exit(EXIT_FAILURE);
    }
    else{
        temp[resp] = '\0';
        printf("%s", temp);
    }
}

void recv_mail()
{
    const char* host_name = "pop.qq.com"; // Specify the mail server domain name
    const unsigned short port = htons(110); // POP3 server port
    const char* user = "858988682@qq.com"; // Specify the user
    const char* pass = "****************"; // Specify the password
    char dest_ip[16];
    int s_fd; // socket file descriptor
    struct hostent *host;
    struct in_addr **addr_list;
    int i = 0;
    int r_size;

    // Get IP from domain name
    if ((host = gethostbyname(host_name)) == NULL)
    {
        herror("gethostbyname");
        exit(EXIT_FAILURE);
    }

    addr_list = (struct in_addr **) host->h_addr_list;
    while (addr_list[i] != NULL)
        ++i;
    strcpy(dest_ip, inet_ntoa(*addr_list[i-1]));

    // Create a socket,return the file descriptor to s_fd
    s_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(s_fd == -1){
        perror("createsocket");
        exit(EXIT_FAILURE);
    }

    // Establish a TCP connection to the POP3 server
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = port; 
    for (int i = 0; i < 8; i ++) server_addr.sin_zero[i] = 0;
    struct in_addr sock_in_addr;
    sock_in_addr.s_addr = inet_addr(dest_ip);
    server_addr.sin_addr = sock_in_addr;
    int addr_len = sizeof(server_addr);
    connect(s_fd, &server_addr, addr_len);

    // Print welcome message
    if ((r_size = recv(s_fd, buf, MAX_SIZE, 0)) == -1)
    {
        perror("recv");
        exit(EXIT_FAILURE);
    }
    buf[r_size] = '\0'; // Do not forget the null terminator
    printf("%s", buf);

    // Send user and password and print server response
    const char* USER = "user ";
    const char* END = "\r\n";
    send(s_fd, USER, strlen(USER), 0);
    send(s_fd, user, strlen(user), 0);
    send(s_fd, END, strlen(END), 0);
    int resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    const char* PASS = "pass ";
    send(s_fd, PASS, strlen(PASS), 0);
    send(s_fd, pass, strlen(pass), 0);
    send(s_fd, END, strlen(END), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send STAT command and print server response
    const char* STAT = "stat\r\n";
    send(s_fd, STAT, strlen(STAT), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);

    // Send LIST command and print server response
    const char* LIST = "list\r\n";
    send(s_fd, LIST, strlen(LIST), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);

    // Retrieve the first mail and print its content
    const char* RET = "retr 1\r\n";
    send(s_fd, RET, strlen(RET), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send QUIT command and print server response
    const char* QUIT = "quit\r\n";
    send(s_fd, QUIT, strlen(QUIT), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    close(s_fd);
}

int main(int argc, char* argv[])
{
    recv_mail();
    exit(0);
}
