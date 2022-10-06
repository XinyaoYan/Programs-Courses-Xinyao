#include "udp.h"
#include "ip.h"
#include "icmp.h"

/**
 * @brief udp处理程序表
 * 
 */
map_t udp_table;

/**
 * @brief udp伪校验和计算
 * 
 * @param buf 要计算的包
 * @param src_ip 源ip地址
 * @param dst_ip 目的ip地址
 * @return uint16_t 伪校验和
 */
static uint16_t udp_checksum(buf_t *buf, uint8_t *src_ip, uint8_t *dst_ip)
{
    int flag = 0;
    // 将被UDP伪头部覆盖的IP头部拷贝出来暂存
    uint8_t src_ip_temp[NET_IP_LEN];
    uint8_t dst_ip_temp[NET_IP_LEN];
    memcpy(src_ip_temp, src_ip, NET_IP_LEN);
    memcpy(dst_ip_temp, dst_ip, NET_IP_LEN);
    
    // 调用函数增加UDP伪头部
    buf_add_header(buf, sizeof(udp_peso_hdr_t));
    
    // 填写UDP伪头部的12字节字段
    udp_peso_hdr_t *p = (udp_peso_hdr_t*)buf -> data;
    memcpy(p -> src_ip, src_ip, NET_IP_LEN);
    memcpy(p -> dst_ip, dst_ip, NET_IP_LEN);
    p -> placeholder = 0;
    p -> protocol = NET_PROTOCOL_UDP;
    p -> total_len16 = swap16(buf -> len - sizeof(udp_peso_hdr_t));
    if(buf -> len % 2 != 0){
        buf_add_padding(buf, 1);
        flag = 1;
    }
    
    // 计算UDP校验和 
    uint16_t checksum = checksum16((uint16_t*)buf -> data, buf -> len / 2);
    
    // 将暂存的IP头部拷贝回来 
    memcpy(src_ip, src_ip_temp, NET_IP_LEN);
    memcpy(dst_ip, dst_ip_temp, NET_IP_LEN);
    
    // 调用buf_remove_header()函数去掉UDP伪头部 
    buf_remove_header(buf, sizeof(udp_peso_hdr_t));
    if(flag){
        buf_remove_padding(buf, 1);
    }

    return swap16(checksum);
}

/**
 * @brief 处理一个收到的udp数据包
 * 
 * @param buf 要处理的包
 * @param src_ip 源ip地址
 */
void udp_in(buf_t *buf, uint8_t *src_ip)
{   
    // 做报头检查，该数据报的长度是否大于等于UDP首部长度和UDP首部长度字段
    udp_hdr_t *p = (udp_hdr_t*)buf -> data;
    if(buf -> len < sizeof(udp_hdr_t))              return;
    if(buf -> len < swap16(p -> total_len16))       return;
    
    // 重新计算校验和
    uint16_t checksum = p -> checksum16;
    uint8_t dst_ip[NET_IP_LEN] = NET_IF_IP;
    p -> checksum16 = 0;
    uint16_t temp_checksum = udp_checksum(buf, src_ip, dst_ip);
    // 如果与接收到的UDP数据报的校验和不一致，丢弃不处理
    if(checksum != temp_checksum)                   return;
    
    // 调用map_get()函数查询udp_table是否有该目的端口号对应的处理函数
    uint16_t port = swap16(p -> dst_port16);
    udp_handler_t* handler = map_get(&udp_table, &port);
    
    // 如果找到，则去掉UDP报头，调用处理函数来做相应的处理
    if(handler){
        buf_remove_header(buf, sizeof(udp_hdr_t));
        (*handler)(buf -> data, buf -> len, src_ip, swap16(p -> src_port16));
    } else{ // 如果找到
        // 调用buf_add_header()函数增加IPv4数据报头部
        buf_add_header(buf, sizeof(udp_hdr_t));
        // 调用icmp_unreachable()函数发送一个端口不可达的差错报文
        icmp_unreachable(buf, src_ip, ICMP_CODE_PORT_UNREACH);
    }
}

/**
 * @brief 处理一个要发送的数据包
 * 
 * @param buf 要处理的包
 * @param src_port 源端口号
 * @param dst_ip 目的ip地址
 * @param dst_port 目的端口号
 */
void udp_out(buf_t *buf, uint16_t src_port, uint8_t *dst_ip, uint16_t dst_port)
{
    // 调用buf_add_header()函数添加UDP报头    
    buf_add_header(buf, sizeof(udp_hdr_t));
    
    // 填充UDP首部字段
    udp_hdr_t *p = (udp_hdr_t*)buf -> data;
    p -> src_port16 = swap16(src_port);
    p -> dst_port16 = swap16(dst_port);
    p -> total_len16 = swap16(buf -> len);
    
    // 计算UDP数据报的校验和
    p -> checksum16 = 0;
    uint8_t src_ip[NET_IP_LEN] = NET_IF_IP;
    p -> checksum16 = udp_checksum(buf, src_ip, dst_ip);

    // 调用ip_out()函数发送UDP数据报
    ip_out(buf, dst_ip, NET_PROTOCOL_UDP);    
}

/**
 * @brief 初始化udp协议
 * 
 */
void udp_init()
{
    map_init(&udp_table, sizeof(uint16_t), sizeof(udp_handler_t), 0, 0, NULL);
    net_add_protocol(NET_PROTOCOL_UDP, udp_in);
}

/**
 * @brief 打开一个udp端口并注册处理程序
 * 
 * @param port 端口号
 * @param handler 处理程序
 * @return int 成功为0，失败为-1
 */
int udp_open(uint16_t port, udp_handler_t handler)
{
    return map_set(&udp_table, &port, &handler);
}

/**
 * @brief 关闭一个udp端口
 * 
 * @param port 端口号
 */
void udp_close(uint16_t port)
{
    map_delete(&udp_table, &port);
}

/**
 * @brief 发送一个udp包
 * 
 * @param data 要发送的数据
 * @param len 数据长度
 * @param src_port 源端口号
 * @param dst_ip 目的ip地址
 * @param dst_port 目的端口号
 */
void udp_send(uint8_t *data, uint16_t len, uint16_t src_port, uint8_t *dst_ip, uint16_t dst_port)
{
    buf_init(&txbuf, len);
    memcpy(txbuf.data, data, len);
    udp_out(&txbuf, src_port, dst_ip, dst_port);
}