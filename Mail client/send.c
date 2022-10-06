#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <getopt.h>
#include "base64_utils.h"
#include "cencode.h"

#define MAX_SIZE 4095

char buf[MAX_SIZE + 1];

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

// receiver: mail address of the recipient
// subject: mail subject
// msg: MESSAGE of mail body or path to the file containing mail body
// att_path: path to the attachment
void send_mail(const char* receiver, const char* subject, const char* msg, const char* att_path)
{
    const char end_msg[] = "\r\n.\r\n";
    const char* host_name = "smtp.qq.com";                  // Specify the mail server domain name
    const unsigned short port = htons(25);                  // SMTP server port
    const char user[] = "ODU4OTg4NjgyQHFxLmNvbQ==\r\n";     // Specify the user
    const char pass[] = "**********************==\r\n";     // Specify the password
    const char from[] = "858988682@qq.com";                 // Specify the mail address of the sender
    char dest_ip[16];                                       // Mail server IP address
    int s_fd;                                               // socket file descriptor
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

    // Create a socket, return the file descriptor to s_fd
    s_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(s_fd == -1){
        perror("createsocket");
        exit(EXIT_FAILURE);
    }

    // Establish a TCP connection to the mail server
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = port;
    for (int i = 0; i < 8; i++){
        server_addr.sin_zero[i] = 0;
    }
    struct in_addr socket_in_addr;
    socket_in_addr.s_addr = inet_addr(dest_ip);
    server_addr.sin_addr = socket_in_addr;
    connect(s_fd, &server_addr, sizeof(server_addr));

    // Print welcome message
    if ((r_size = recv(s_fd, buf, MAX_SIZE, 0)) == -1)
    {
        perror("recv");
        exit(EXIT_FAILURE);
    }
    buf[r_size] = '\0'; // Do not forget the null terminator
    printf("%s", buf);

    // Send EHLO command
    const char EHLO[] = "EHLO qq.com\r\n";
    send(s_fd, EHLO, strlen(EHLO), 0);
    // Print server response
    int resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Authentication. Server response should be printed out.
    const char AUTH[] = "AUTH login\r\n";
    send(s_fd, AUTH, strlen(AUTH), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);

    // User and pass
    send(s_fd, user, strlen(user), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    send(s_fd, pass, strlen(pass), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send MAIL FROM command and print server response
    const char MAILFROM[] = "MAIL FROM:<858988682@qq.com>\r\n";
    send(s_fd, MAILFROM, strlen(MAILFROM), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send RCPT TO command and print server response
    const char RCPT[] = "RCPT TO:<";
    send(s_fd, RCPT, strlen(RCPT), 0);
    send(s_fd, receiver, strlen(receiver), 0);
    const char* RCPT_END = ">\r\n";
    send(s_fd, RCPT_END, strlen(RCPT_END), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send DATA command and print server response
    const char DATA[] = "data\r\n";
    send(s_fd, DATA, strlen(DATA), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    
    // Send message data
    // --- From ---
    const char FROM[] = "from:";
    const char END[] = "\r\n";
    send(s_fd, FROM, strlen(FROM), 0);
    send(s_fd, from, strlen(from), 0);
    send(s_fd, END, strlen(END), 0);
    // --- To ---
    const char TO[] = "to:";
    send(s_fd, TO, strlen(TO), 0);
    send(s_fd, receiver, strlen(receiver), 0);
    send(s_fd, END, strlen(END), 0);
    // --- Subject ---
    const char SUBJECT[] = "subject:";
    send(s_fd, SUBJECT, strlen(SUBJECT), 0);
    send(s_fd, subject, strlen(subject), 0);
    send(s_fd, END, strlen(END), 0);
    // --- Version ---
    const char VERSION[] = "MIME-Version: 1.0\r\n";
    send(s_fd, VERSION, strlen(VERSION), 0);
    // --- Content Type ---
    const char TYPE[] = "Content-Type: multipart/mixed; \
                         boundary='qwertyuiopasdfghjklzxcvbnm'\r\n";
    send(s_fd, TYPE, strlen(TYPE), 0);
    
    // --- Text ---
    const char BOUND[] = "--qwertyuiopasdfghjklzxcvbnm\r\n";
    send(s_fd, BOUND, strlen(BOUND), 0);   
    const char MESSAGE_TYPE[] = "Content-Type: text/plain; charset=gb2312\r\n";
    send(s_fd, MESSAGE_TYPE, strlen(MESSAGE_TYPE), 0);
    send(s_fd, msg, strlen(msg), 0);

    // --- Attach File ---
    send(s_fd, END, strlen(END), 0);
    send(s_fd, BOUND, strlen(BOUND), 0);
    const char ENCODE[] = "Content-Transfer-Encoding: base64\r\n";
    send(s_fd, ENCODE, strlen(ENCODE), 0);
    const char PATH_TYPE[] = "Content-Type: application/octet-stream; name=";
    send(s_fd, PATH_TYPE, strlen(PATH_TYPE), 0);
    send(s_fd, att_path, strlen(att_path), 0);
    send(s_fd, END, strlen(END), 0);

    // --- (base64 content) ---
    FILE* fp = fopen(att_path, "r");
    char* in = (char*)malloc(MAX_SIZE);
    char* encoded = (char*)malloc(2 * MAX_SIZE); 
    base64_encodestate es;
    base64_init_encodestate(&es);
    while(1){
        if (fread(in, sizeof(char), MAX_SIZE, fp) == 0) break;
        char* out = encode_str(in);
        send(s_fd, out, strlen(out), 0);
    }

    // Message ends with a single period
    send(s_fd, end_msg, strlen(end_msg), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);

    // Send QUIT command and print server response
    const char QUIT[] = "quit\n";
    send(s_fd, QUIT, strlen(QUIT), 0);
    resp = recv(s_fd, buf, MAX_SIZE, 0);
    recv_resp_info(resp, buf);
    close(s_fd);
}

int main(int argc, char* argv[])
{
    int opt;
    char* s_arg = NULL;
    char* m_arg = NULL;
    char* a_arg = NULL;
    char* recipient = NULL;
    const char* optstring = ":s:m:a:";
    while ((opt = getopt(argc, argv, optstring)) != -1)
    {
        switch (opt)
        {
        case 's':
            s_arg = optarg;
            break;
        case 'm':
            m_arg = optarg;
            break;
        case 'a':
            a_arg = optarg;
            break;
        case ':':
            fprintf(stderr, "Option %c needs an argument.\n", optopt);
            exit(EXIT_FAILURE);
        case '?':
            fprintf(stderr, "Unknown option: %c.\n", optopt);
            exit(EXIT_FAILURE);
        default:
            fprintf(stderr, "Unknown error.\n");
            exit(EXIT_FAILURE);
        }
    }

    if (optind == argc)
    {
        fprintf(stderr, "Recipient not specified.\n");
        exit(EXIT_FAILURE);
    }
    else if (optind < argc - 1)
    {
        fprintf(stderr, "Too many arguments.\n");
        exit(EXIT_FAILURE);
    }
    else
    {
        recipient = argv[optind];
        send_mail(recipient, s_arg, m_arg, a_arg);
        exit(0);
    }
}
