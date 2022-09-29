#include <string.h>
#include <stdio.h>
#include "net.h"
#include "arp.h"
#include "ethernet.h"
/**
 * @brief 初始的arp包
 * 
 */
static const arp_pkt_t arp_init_pkt = {
    .hw_type16 = swap16(ARP_HW_ETHER), 
    .pro_type16 = swap16(NET_PROTOCOL_IP), 
    .hw_len = NET_MAC_LEN, 
    .pro_len = NET_IP_LEN, 
    .sender_ip = NET_IF_IP, 
    .sender_mac = NET_IF_MAC, 
    .target_mac = {0}};

/**
 * @brief arp地址转换表，<ip, mac>的容器
 * 
 */
map_t arp_table;

/**
 * @brief arp buffer，<ip, buf_t>的容器
 * 
 */
map_t arp_buf;

/**
 * @brief 打印一条arp表项
 * 
 * @param ip 表项的ip地址
 * @param mac 表项的mac地址
 * @param timestamp 表项的更新时间
 */
void arp_entry_print(void *ip, void *mac, time_t *timestamp)
{
    printf("%s | %s | %s\n", iptos(ip), mactos(mac), timetos(*timestamp));
}

/**
 * @brief 打印整个arp表
 * 
 */
void arp_print()
{
    printf(" ===ARP TABLE BEGIN ===\n");
    map_foreach(&arp_table, arp_entry_print);
    printf(" ===ARP TABLE  END  ===\n");
}

/**
 * @brief 发送一个arp请求
 * 
 * @param target_ip 想要知道的目标的ip地址
 */
void arp_req(uint8_t *target_ip)
{
    // 调用buf_init()对txbuf进行初始化
    buf_init(&txbuf, sizeof(arp_pkt_t));

    // 填写ARP报头
    arp_pkt_t *p = (arp_pkt_t*)txbuf.data;
    p -> hw_type16 = swap16(ARP_HW_ETHER);
    p -> pro_type16 = swap16(NET_PROTOCOL_IP);
    p -> hw_len = NET_MAC_LEN;
    p -> pro_len = NET_IP_LEN;
    p -> opcode16 = swap16(ARP_REQUEST);  // ARP操作类型为ARP_REQUEST，注意大小端转换

    // 填写发送方和接收方的MAC地址和IP地址
    uint8_t store_mac[NET_MAC_LEN] = NET_IF_MAC;
    uint8_t store_ip[NET_IP_LEN] = NET_IF_IP;
    memcpy(p -> sender_mac, store_mac, NET_MAC_LEN);
    memcpy(p -> sender_ip, store_ip, NET_IP_LEN);
    memset(p -> target_mac, 0, NET_MAC_LEN);
    memcpy(p -> target_ip, target_ip, NET_IP_LEN);

    // 调用ethernet_out函数将ARP报文发送出去
    ethernet_out(&txbuf, ether_broadcast_mac, NET_PROTOCOL_ARP);

}

/**
 * @brief 发送一个arp响应
 * 
 * @param target_ip 目标ip地址
 * @param target_mac 目标mac地址
 */
void arp_resp(uint8_t *target_ip, uint8_t *target_mac)
{
    // 调用buf_init()来初始化txbuf
    buf_init(&txbuf, sizeof(arp_pkt_t));
    arp_pkt_t *p = (arp_pkt_t*)txbuf.data;

    // 填写ARP报文头部
    p -> hw_type16 = swap16(ARP_HW_ETHER);
    p -> pro_type16 = swap16(NET_PROTOCOL_IP);
    p -> hw_len = NET_MAC_LEN;
    p -> pro_len = NET_IP_LEN;
    p -> opcode16 = swap16(ARP_REPLY);
    
    // 填写发送方和接收方的MAC地址和IP地址
    uint8_t store_mac[NET_MAC_LEN] = NET_IF_MAC;
    uint8_t store_ip[NET_IP_LEN] = NET_IF_IP;
    memcpy(p -> sender_mac, store_mac, NET_MAC_LEN);
    memcpy(p -> sender_ip, store_ip, NET_IP_LEN);
    memcpy(p -> target_mac, target_mac, NET_MAC_LEN);
    memcpy(p -> target_ip, target_ip, NET_IP_LEN);

    // 调用ethernet_out()函数将填好的ARP报文发出去
    ethernet_out(&txbuf, target_mac, NET_PROTOCOL_ARP);
}

/**
 * @brief 处理一个收到的数据包
 * 
 * @param buf 要处理的数据包
 * @param src_mac 源mac地址
 */
void arp_in(buf_t *buf, uint8_t *src_mac)
{
    // 首先判断数据长度，如果数据长度小于ARP头部长度，则丢弃不处理
    if(buf -> len<sizeof(arp_pkt_t))      return;

    // 做报头检查，查看报文是否完整，检测内容包括：
    arp_pkt_t *p = (arp_pkt_t*)buf -> data;
    if(swap16(p -> hw_type16) == ARP_HW_ETHER           // ARP报头的硬件类型
        && swap16(p -> pro_type16) == NET_PROTOCOL_IP   // 上层协议类型
        && p -> hw_len == NET_MAC_LEN                   // MAC硬件地址长度
        && p -> pro_len == NET_IP_LEN                   // IP协议地址长度
        && (swap16(p -> opcode16) == ARP_REQUEST        // 报头是否符合协议规定
        || swap16(p -> opcode16) == ARP_REPLY)){
        
        uint8_t *sender_ip = p -> sender_ip;
        uint8_t *target_ip = p -> target_ip;

        // 调用map_set()函数更新ARP表项
        map_set(&arp_table, sender_ip, src_mac);

        // 调用map_set()函数查看该接收报文的IP地址是否有对应的arp_buf缓存
        buf_t *buffer = map_get(&arp_buf, sender_ip);
        
        // 如果有，则 
        if(buffer != NULL){
            // 调用ethernet_out()函数直接转发缓存的数据包arp_buf
            ethernet_out(buffer, src_mac, NET_PROTOCOL_IP);
            // 调用map_delete()函数将这个缓存的数据包删掉
            map_delete(&arp_buf, sender_ip);
        }

        // 如果没有，则
        else{
            // 判断接收到的报文是否为ARP_REQUEST
            if(swap16(p -> opcode16) == ARP_REQUEST){
                uint8_t store_ip[NET_IP_LEN] = NET_IF_IP;
                // 再判断该请求报文的target_ip是否是本机的IP
                if(memcmp(target_ip, store_ip, NET_IP_LEN) == 0)
                    // 调用arp_resp()函数回应一个响应报文
                    arp_resp(sender_ip, src_mac);
            }
        }
    }
}

/**
 * @brief 处理一个要发送的数据包
 * 
 * @param buf 要处理的数据包
 * @param ip 目标ip地址
 * @param protocol 上层协议
 */
void arp_out(buf_t *buf, uint8_t *ip)
{
    // 调用map_get()函数，根据IP地址查找ARP表
    uint8_t *target_mac = map_get(&arp_table, ip);

    // 如果能找到该IP对应的MAC地址
    if(target_mac != NULL)
        // 将数据包直接发送给以太网层
        ethernet_out(buf, target_mac, NET_PROTOCOL_IP);
    
    else{
        // 判断arp_buf是否已经有包
        buf_t *buffer = map_get(&arp_buf, ip);

        // 如果有则不能再发送ARP请求
        if(buffer != NULL) return;
        else{
            // 调用map_set()函数将来自IP层的数据包缓存到arp_buf
            map_set(&arp_buf, ip, buf);
            // 调用arp_req()函数，发送一个和请求目标IP地址对应的MAC地址的ARP报文
            arp_req(ip);
        }
    } 
}

/**
 * @brief 初始化arp协议
 * 
 */
void arp_init()
{
    map_init(&arp_table, NET_IP_LEN, NET_MAC_LEN, 0, ARP_TIMEOUT_SEC, NULL);
    map_init(&arp_buf, NET_IP_LEN, sizeof(buf_t), 0, ARP_MIN_INTERVAL, buf_copy);
    net_add_protocol(NET_PROTOCOL_ARP, arp_in);
    arp_req(net_if_ip);
}