#ifndef _TYPES_H_
#define _TYPES_H_

/*****************************************************************************
* SECTION: Type def
******************************************************************************/
typedef int boolean;
typedef uint16_t flag16;

typedef enum newfs_file_type
{
    NEWFS_REG_FILE,     // 普通文件
    NEWFS_DIR           // 目录文件
} NEWFS_FILE_TYPE;

/*****************************************************************************
* SECTION: Macro
******************************************************************************/
#define TRUE 1
#define FALSE 0
#define UINT32_BITS 32
#define UINT8_BITS 8

#define NEWFS_MAGIC_NUM 0x52415453
#define NEWFS_SUPER_OFS 0
#define NEWFS_ROOT_INO 0

#define NEWFS_ERROR_NONE 0
#define NEWFS_ERROR_ACCESS EACCES
#define NEWFS_ERROR_SEEK ESPIPE
#define NEWFS_ERROR_ISDIR EISDIR
#define NEWFS_ERROR_NOSPACE ENOSPC
#define NEWFS_ERROR_EXISTS EEXIST
#define NEWFS_ERROR_NOTFOUND ENOENT
#define NEWFS_ERROR_UNSUPPORTED ENXIO
#define NEWFS_ERROR_IO EIO       // Error Input/Output 
#define NEWFS_ERROR_INVAL EINVAL // Invalid Args 

#define MAX_NAME_LEN 128
#define NEWFS_INODE_PER_FILE 1
#define NEWFS_DATA_PER_FILE 16
#define NEWFS_DEFAULT_PERM 0777

#define NEWFS_IOC_MAGIC 'S'
#define NEWFS_IOC_SEEK _IO(NEWFS_IOC_MAGIC, 0)

#define NEWFS_FLAG_BUF_DIRTY 0x1
#define NEWFS_FLAG_BUF_OCCUPY 0x2
/*****************************************************************************
* SECTION: Macro Function
******************************************************************************/
#define NEWFS_IO_SZ() (newfs_super.sz_io)
#define NEWFS_DISK_SZ() (newfs_super.sz_disk)
#define NEWFS_DRIVER() (newfs_super.fd)

#define NEWFS_ROUND_DOWN(value, round) (value % round == 0 ? value : (value / round) * round)
#define NEWFS_ROUND_UP(value, round) (value % round == 0 ? value : (value / round + 1) * round)

#define NEWFS_BLKS_SZ(blks) (blks * NEWFS_IO_SZ())
#define NEWFS_ASSIGN_FNAME(pnewfs_dentry, _fname) memcpy(pnewfs_dentry->fname, _fname, strlen(_fname))
#define NEWFS_INO_OFS(ino) (newfs_super.data_offset + ino * NEWFS_BLKS_SZ(( \
                                                            NEWFS_INODE_PER_FILE + NEWFS_DATA_PER_FILE)))
#define NEWFS_DATA_OFS(ino) (NEWFS_INO_OFS(ino) + NEWFS_BLKS_SZ(NEWFS_INODE_PER_FILE))

#define NEWFS_IS_DIR(pinode) (pinode->dentry->ftype == NEWFS_DIR)
#define NEWFS_IS_REG(pinode) (pinode->dentry->ftype == NEWFS_REG_FILE)

/*****************************************************************************
* SECTION: FS Specific Structure - In memory structure
******************************************************************************/
struct newfs_dentry;
struct newfs_inode;
struct newfs_super;

struct custom_options
{
    const char *device;
    boolean show_help;
};

struct newfs_inode
{
    int ino;                        // 在inode位图中的下标 
    int size;                       // 文件已占用空间 
    int dir_cnt;                    // 若为目录文件，目录项的数量 
    struct newfs_dentry *dentry;    // 指向该inode的目录项 
    struct newfs_dentry *dentrys;   // 所有目录项 
    int block_p[6];                 // 指向保存数据的数据块 
};

struct newfs_dentry
{
    struct newfs_inode *inode;      // 指向对应的索引项inode 
    int ino;                        // 指向的inode文件号 
    char fname[MAX_NAME_LEN];       // 指向的inode文件名
    NEWFS_FILE_TYPE ftype;          // 指向的inode文件类型 
    struct newfs_dentry *parent;    // 父亲inode的dentry 
    struct newfs_dentry *brother;   // 兄弟inode的dentry 
};

struct newfs_super
{   
    // 基本信息
    int fd;                             // 文件描述符
    int sz_io;                          // IO的大小
    int sz_disk;                        // disk的大小
    int sz_usage;
    int max_ino;                        // 最多支持的文件数

    // inode块和数据块信息
    int inode_offset;                   // inode块在磁盘上的偏移量
    int inode_blks;                     // inode位图占用的磁盘块数

    // 位图信息
    uint8_t *map_inode;                 // 指向inode位图
    uint8_t *map_data;                  // 指向data位图
    int map_inode_offset;               // inode位图在磁盘上的偏移
    int map_data_offset;                // data位图在磁盘上的偏移
    int data_offset;                    // data块在磁盘上的偏移量
    int data_blks;                      // data位图占用的磁盘块数

    boolean is_mounted;                 // 是否挂载
    struct newfs_dentry *root_dentry;   // 指向根目录
};

static inline struct newfs_dentry *new_dentry(char *fname, NEWFS_FILE_TYPE ftype)
{
    struct newfs_dentry *dentry = (struct newfs_dentry *)malloc(sizeof(struct newfs_dentry));
    memset(dentry, 0, sizeof(struct newfs_dentry));
    NEWFS_ASSIGN_FNAME(dentry, fname);
    dentry->ftype = ftype;
    dentry->ino = -1;
    dentry->inode = NULL;
    dentry->parent = NULL;
    dentry->brother = NULL;
}

/*****************************************************************************
* SECTION: FS Specific Structure - Disk structure
******************************************************************************/
struct newfs_super_d
{
    uint32_t magic_num;         // 幻数
    int sz_usage;               // 使用的大小
    int max_ino;                // 最多支持的文件数
    int map_inode_offset;       // inode位图在磁盘上的偏移
    int map_data_offset;        // data位图在磁盘上的偏移
    int inode_blks;             // inode位图占用的磁盘块数
    int data_blks;              // data位图占用的磁盘块数
    int inode_offset;           // inode块的偏移
    int data_offset;            // data块的偏移
};

struct newfs_inode_d
{
    int ino;                    // 在inode位图中的下标 
    int size;                   // 文件已占用空间 
    int dir_cnt;                // 目录项的数量
    int block_p[6];             // 指向保存数据的数据块 
    NEWFS_FILE_TYPE ftype;      // 文件类型
};

struct newfs_dentry_d
{
    char fname[MAX_NAME_LEN];   // 指向的inode文件名
    NEWFS_FILE_TYPE ftype;      // 指向的inode文件类型
    int ino;                    // 指向的inode编号
};

#endif // _TYPES_H_ 