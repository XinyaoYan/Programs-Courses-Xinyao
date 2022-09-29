#include "net.h"
#include "icmp.h"
#include "ip.h"

/**
 * @brief 发送icmp响应
 * 
 * @param req_buf 收到的icmp请求包
 * @param src_ip 源ip地址
 */
static void icmp_resp(buf_t *req_buf, uint8_t *src_ip)
{
    // 调用buf_init()来初始化txbuf
    buf_init(&txbuf, req_buf -> len);
    
    // 封装包头
    icmp_hdr_t* p = (icmp_hdr_t *)req_buf -> data;
    p -> type = ICMP_TYPE_ECHO_REPLY;
    p -> code = 0;
    p -> checksum16 = 0;
    
    // 初始化报文数据
    memcpy(txbuf.data, req_buf -> data, req_buf -> len);
    // 填写报文数据
    memcpy(txbuf.data, p, sizeof(icmp_hdr_t));
    // 计算校验和
    p -> checksum16 = swap16(checksum16((uint16_t*)txbuf.data, req_buf -> len / 2));
    // 更新报文数据
    memcpy(txbuf.data, p, sizeof(icmp_hdr_t));
    
    // 调用ip_out()函数将数据报发送出去
    ip_out(&txbuf, src_ip, NET_PROTOCOL_ICMP);
}

/**
 * @brief 处理一个收到的数据包
 * 
 * @param buf 要处理的数据包
 * @param src_ip 源ip地址
 */
void icmp_in(buf_t *buf, uint8_t *src_ip)
{
    // 如果数据包的长度小于ICMP头部长度，丢弃不处理
    if(buf -> len < sizeof(icmp_hdr_t))      return;

    // 查看报文的ICMP类型是否为回显请求
    icmp_hdr_t *p = (icmp_hdr_t*)buf -> data;
    if(p -> type == ICMP_TYPE_ECHO_REQUEST){
        // 调用icmp_resp()函数回送一个ping应答
        icmp_resp(buf, src_ip);
    }
}

/**
 * @brief 发送icmp不可达
 * 
 * @param recv_buf 收到的ip数据包
 * @param src_ip 源ip地址
 * @param code icmp code，协议不可达或端口不可达
 */
void icmp_unreachable(buf_t *recv_buf, uint8_t *src_ip, icmp_code_t code)
{   
    // 调用buf_init()函数来初始化txbuf
    buf_init(&txbuf, sizeof(icmp_hdr_t) + sizeof(ip_hdr_t) + 8);

    // 填写ICMP报头首部
    txbuf.data[0] = ICMP_TYPE_UNREACH;
    txbuf.data[1] = code;
    for(int i = 2; i < 8; i++){
        txbuf.data[i] = 0; // 校验和、标识符、序列号都填0
    }

    // 复制IP数据报首部和IP数据报的前8个字节的数据字段
    memcpy(&txbuf.data[8], recv_buf->data, sizeof(ip_hdr_t) + 8); 

    // 填写校验和
    uint16_t checksum = checksum16((uint16_t *)txbuf.data, txbuf.len / 2);
    txbuf.data[2] = (checksum & 0xff00)>>8;
    txbuf.data[3] = checksum & 0x00ff; 
    
    // 调用ip_out()函数将数据报发送出去
    ip_out(&txbuf, src_ip, NET_PROTOCOL_ICMP);
}

/**
 * @brief 初始化icmp协议
 * 
 */
void icmp_init(){
    net_add_protocol(NET_PROTOCOL_ICMP, icmp_in);
}