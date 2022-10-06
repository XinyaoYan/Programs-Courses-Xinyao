#include "net.h"
#include "ip.h"
#include "ethernet.h"
#include "arp.h"
#include "icmp.h"
#include "config.h"

uint32_t id = 0;

/**
 * @brief 处理一个收到的数据包
 * 
 * @param buf 要处理的数据包
 * @param src_mac 源mac地址
 */
void ip_in(buf_t *buf, uint8_t *src_mac)
{   
    // 如果数据包的长度小于IP头部长度，丢弃不处理
    if(buf -> len < (sizeof(ip_hdr_t) / IP_HDR_LEN_PER_BYTE))      return;

    // 做报头检查，包括IP头部版本号和头部长度
    ip_hdr_t *p = (ip_hdr_t*)buf -> data;
    if(p -> version != IP_VERSION_4 || p -> hdr_len > buf -> len)   return;
    
    // 先把IP头部校验和用其他变量保存起来
    uint16_t temp = (buf->data[10]<<8) + buf->data[11];

    // 计算头部校验和，先置零，如果正确则恢复，否则丢弃
    buf->data[10] = buf->data[11] = 0;
    uint16_t cksum = checksum16((uint16_t *)buf->data, p -> hdr_len * 2);
    if(cksum != temp) return;
    buf->data[10] = (uint8_t)((cksum&0xff00) >> 8);
    buf->data[11] = (uint8_t)(cksum&0x00ff);

    // 如果目的IP地址不是本机的IP地址，丢弃不处理
    uint8_t store_ip[NET_IP_LEN] = NET_IF_IP;
    if(memcmp(p -> dst_ip, store_ip, NET_IP_LEN) != 0)    return;
        
    // 如果接收到的数据包的长度大于IP头部的总长度字段
    if(buf -> len > p -> total_len16) 
        // 调用buf_remove_padding()函数去除填充字段
        buf_remove_padding(buf, buf -> len - p -> total_len16);
        
    // 调用buf_remove_header()函数去掉IP报头
    if(p -> protocol == NET_PROTOCOL_ICMP || p -> protocol == NET_PROTOCOL_UDP){
        buf_remove_header(buf, p -> hdr_len * IP_HDR_LEN_PER_BYTE);
        // 调用net_in()函数向上层传递数据包
        net_in(buf, p -> protocol, p -> src_ip);
    }
    else{   // ICMP不可达
        icmp_unreachable(buf, p -> src_ip, ICMP_CODE_PROTOCOL_UNREACH);
    }
}

/**
 * @brief 处理一个要发送的ip分片
 * 
 * @param buf 要发送的分片
 * @param ip 目标ip地址
 * @param protocol 上层协议
 * @param id 数据包id
 * @param offset 分片offset，必须被8整除
 * @param mf 分片mf标志，是否有下一个分片
 */
void ip_fragment_out(buf_t *buf, uint8_t *ip, net_protocol_t protocol, int id, uint16_t offset, int mf)
{   
    // 调用buf_add_header()函数增加IP数据报头空间
    buf_add_header(buf, sizeof(ip_hdr_t));
    ip_hdr_t *p = (ip_hdr_t*)buf -> data;

    // 填写IP报文头部
    p -> hdr_len = 5;
    p -> version = IP_VERSION_4;
    p -> tos = 0;
    p -> total_len16 = swap16(buf -> len);
    p -> id16 = swap16(id);
    p -> flags_fragment16 = (mf << 5) + ((offset & 0x1f00) >> 8) + ((offset & 0x00ff) << 8);
    p -> ttl = 64;
    p -> protocol = protocol;
    p -> hdr_checksum16 = 0;

    // 填写发送方和接收方的IP地址
    uint8_t store_ip[NET_IP_LEN] = NET_IF_IP;
    memcpy(p -> src_ip, store_ip, NET_IP_LEN);
    memcpy(p -> dst_ip, ip, NET_IP_LEN);

    // 调用checksum16()函数计算校验和
    p -> hdr_checksum16 = swap16(checksum16((uint16_t*)p, 10));

    // 调用arp_out()函数将填好的IP报文发出去
    arp_out(buf, ip);
}

/**
 * @brief 处理一个要发送的ip数据包
 * 
 * @param buf 要处理的包
 * @param ip 目标ip地址
 * @param protocol 上层协议
 */
void ip_out(buf_t *buf, uint8_t *ip, net_protocol_t protocol)
{   
    // 计算分片的数量
    int slice_num = (buf -> len % 1480 == 0) ? buf -> len / 1480 : (int)(buf -> len / 1480) + 1;
    
    // 需要分片
    if(slice_num > 1){
        // 对于每一个分片(除最后一片)
        for(int i = 0; i < slice_num - 1; i++){
            buf_t ip_buf;
            buf_init(&ip_buf, 1480);    // 包长 = IP协议最大负载包长
            memcpy(ip_buf.data, &buf -> data[i * 1480], 1480);
            ip_fragment_out(&ip_buf, ip, protocol, id, i * 1480 / IP_HDR_OFFSET_PER_BYTE, 1);
        }
        // 对于最后一个分片
        int offset = (slice_num - 1) * 1480;
        buf_t ip_buf;
        buf_init(&ip_buf, buf -> len - offset);     // 包长等于该分片大小
        memcpy(ip_buf.data, &buf -> data[offset], buf -> len - offset);
        ip_fragment_out(&ip_buf, ip, protocol, id, offset / IP_HDR_OFFSET_PER_BYTE, 0);
    }

    // 不需要分片
    else{
        ip_fragment_out(buf, ip, protocol, id, 0, 0);
    }
    id += 1;
}

/**
 * @brief 初始化ip协议
 * 
 */
void ip_init()
{
    net_add_protocol(NET_PROTOCOL_IP, ip_in);
}