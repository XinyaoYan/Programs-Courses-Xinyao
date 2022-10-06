#include "ethernet.h"
#include "utils.h"
#include "driver.h"
#include "arp.h"
#include "ip.h"

/**
 * @brief 处理一个收到的数据包
 * 
 * @param buf 要处理的数据包
 */
void ethernet_in(buf_t *buf)
{   
    if(buf->len < sizeof(ether_hdr_t))  return;
    
    // 调用buf_remove_header()函数移除加以太网包头
    ether_hdr_t header;
    memcpy(&header, buf->data, sizeof(ether_hdr_t));
    buf_remove_header(buf, sizeof(ether_hdr_t));
    
    // 调用net_in()函数向上层传递数据包, protocol需要反转大小端
    net_in(buf, swap16(header.protocol16), header.src);
}
/**
 * @brief 处理一个要发送的数据包
 * 
 * @param buf 要处理的数据包
 * @param mac 目标MAC地址
 * @param protocol 上层协议
 */
void ethernet_out(buf_t *buf, const uint8_t *mac, net_protocol_t protocol)
{
    // 首先判断数据长度，如果不足46则显式填充0，填充可以调用buf_add_padding()函数来实现
    if(buf->len < 46)
        buf_add_padding(buf, (size_t)(46 - buf->len));
    
    // 调用buf_add_header()函数添加以太网包头
    buf_add_header(buf, sizeof(ether_hdr_t));
    ether_hdr_t *hdr = (ether_hdr_t *)buf->data;
    
    // 填写目的MAC地址
    memcpy(hdr->dst, mac, NET_MAC_LEN * sizeof(uint8_t)); 
    
    // 填写源MAC地址，即本机的MAC地址
    memcpy(hdr->src, net_if_mac, NET_MAC_LEN * sizeof(uint8_t));
    
    // 填写协议类型 protocol
    uint16_t protocol16 = swap16(protocol);     // 翻转大小端
    memcpy(&(hdr->protocol16),  &protocol16, sizeof(uint16_t));
    
    // 调用驱动层封装好的driver_send()发送函数，将添加了以太网包头的数据帧发送到驱动层
    driver_send(buf);
}
/**
 * @brief 初始化以太网协议
 * 
 */
void ethernet_init()
{
    buf_init(&rxbuf, ETHERNET_MAX_TRANSPORT_UNIT + sizeof(ether_hdr_t));
}

/**
 * @brief 一次以太网轮询
 * 
 */
void ethernet_poll()
{
    if (driver_recv(&rxbuf) > 0)
        ethernet_in(&rxbuf);
}
