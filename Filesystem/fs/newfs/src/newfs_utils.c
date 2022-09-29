#include "../include/newfs.h"
#include "../include/types.h"

// get file name
char *newfs_get_fname(const char *path) {
    char ch = '/';
    char *q = strrchr(path, ch) + 1;        // point to the end of the file
    return q;
}

// calculate the level of the path
int newfs_calc_lvl(const char *path) {
    const char *str = path;
    int lvl = 0;
    if (strcmp(path, "/") == 0) {
        return lvl;
    }
    while (*str != NULL) {
        if (*str == '/') {
            lvl++;
        }
        str++;
    }
    return lvl;
}

// read contents from disk according to offset
int newfs_driver_read(int offset, uint8_t *out_content, int size) {   
    // offset and size aligned with sz_io
    int offset_aligned = NEWFS_ROUND_DOWN(offset, NEWFS_IO_SZ());
    int bias = offset - offset_aligned;
    int size_aligned = NEWFS_ROUND_UP((size + bias), NEWFS_IO_SZ());
    
    // allocate space
    uint8_t *temp_content = (uint8_t *)malloc(size_aligned);
    uint8_t *cur = temp_content;    // point to the space
    
    // move read position to offset
    ddriver_seek(NEWFS_DRIVER(), offset_aligned, SEEK_SET);
    while (size_aligned != 0) {
        // read(NEWFS_DRIVER(), cur, NEWFS_IO_SZ());
        ddriver_read(NEWFS_DRIVER(), cur, NEWFS_IO_SZ());
        cur += NEWFS_IO_SZ();
        size_aligned -= NEWFS_IO_SZ();
    }

    // write to out_content
    memcpy(out_content, temp_content + bias, size);
    free(temp_content);
    return NEWFS_ERROR_NONE;
}

// write contents to disk according to offset
int newfs_driver_write(int offset, uint8_t *in_content, int size) {
    int offset_aligned = NEWFS_ROUND_DOWN(offset, NEWFS_IO_SZ());
    int bias = offset - offset_aligned;
    int size_aligned = NEWFS_ROUND_UP((size + bias), NEWFS_IO_SZ());
    uint8_t *temp_content = (uint8_t *)malloc(size_aligned);
    uint8_t *cur = temp_content;

    // read first
    newfs_driver_read(offset_aligned, temp_content, size_aligned);

    // copy content from in_content to temp_content + bias
    memcpy(temp_content + bias, in_content, size);

    // lseek(NEWFS_DRIVER(), offset_aligned, SEEK_SET);
    ddriver_seek(NEWFS_DRIVER(), offset_aligned, SEEK_SET);
    while (size_aligned != 0) {
        // write(NEWFS_DRIVER(), cur, NEWFS_IO_SZ());
        ddriver_write(NEWFS_DRIVER(), cur, NEWFS_IO_SZ());
        cur += NEWFS_IO_SZ();
        size_aligned -= NEWFS_IO_SZ();
    }

    free(temp_content);
    return NEWFS_ERROR_NONE;
}

// allocate an entry to the inode
// use the header insertion method
int newfs_alloc_dentry(struct newfs_inode* inode, struct newfs_dentry* dentry) {
    int byte_cursor, bit_cursor;

    // if there are none denrty under inode
    if (inode->dentrys == NULL) {
        inode->dentrys = dentry;
    }

    else {
        dentry->brother = inode->dentrys;   // define its brother dentry
        inode->dentrys = dentry;            // define the dentry
    }
    inode->dir_cnt++;

    for (int i = 0; i <= (sizeof(struct newfs_dentry_d) * inode->dir_cnt) / NEWFS_IO_SZ(); i++){
        byte_cursor = inode->block_p[i] / UINT8_BITS;
        bit_cursor = inode->block_p[i] % UINT8_BITS;
        newfs_super.map_data[byte_cursor] |= (0x1 << bit_cursor);
    }

    return inode->dir_cnt;  // return the amount of dentries under inode
}

// drop the dentry under the inode
int newfs_drop_dentry(struct newfs_inode * inode, struct newfs_dentry * dentry) {
    boolean is_find = FALSE;
    struct newfs_dentry* dentry_cursor;
    dentry_cursor = inode->dentrys;
    
    // if inode->dentrys = dentry
    if (dentry_cursor == dentry) {
        inode->dentrys = dentry->brother;
        is_find = TRUE;
    }

    else {
        while (dentry_cursor){
            // find the dentry by finding its brother dentry
            if (dentry_cursor->brother == dentry) {
                dentry_cursor->brother = dentry->brother;
                is_find = TRUE;
                break;
            }
            dentry_cursor = dentry_cursor->brother;
        }
    }
    if (!is_find) {
        return -NEWFS_ERROR_NOTFOUND;
    }
    inode->dir_cnt--;
    return inode->dir_cnt;
}

// find an free inode in the inodeblock bitmap and allocate it to an entry
struct newfs_inode* newfs_alloc_inode(struct newfs_dentry * dentry) {
    struct newfs_inode* inode;
    int byte_cursor = 0;    // record the number of bytes that were traversed in the bitmap
    int bit_cursor = 0;     // record the number of bits that were traversed in the byte
    int ino_cursor = 0;     // record the number of bits that are not free
    boolean is_find_free_entry = FALSE;

    // traverse each byte of the bitmap
    for (byte_cursor = 0; byte_cursor < NEWFS_BLKS_SZ(newfs_super.inode_blks); byte_cursor++){
        // traverse each bit of the byte
        for (bit_cursor = 0; bit_cursor < UINT8_BITS; bit_cursor++) {
            if(byte_cursor == 0 && bit_cursor < 2){
                ino_cursor++;
                continue;
            }
            
            // find free entry
            if((newfs_super.map_inode[byte_cursor] & (0x1 << bit_cursor)) == 0) {    
                newfs_super.map_inode[byte_cursor] |= (0x1 << bit_cursor);
                is_find_free_entry = TRUE;           
                break;
            }
            ino_cursor++;
        }

        // if find free entry, end traversal
        if (is_find_free_entry) {
            break;
        }
    }

    // all the entries in the bitmap are not free
    if (!is_find_free_entry || ino_cursor == newfs_super.max_ino)
        return -NEWFS_ERROR_NOSPACE;

    // request memory to store inode
    inode = (struct newfs_inode*)malloc(sizeof(struct newfs_inode));
    inode->ino  = ino_cursor; 
    inode->size = 0;
    
    // dentry points to inode
    dentry->inode = inode;
    dentry->ino   = inode->ino;
    
    // inode points back to dentry
    inode->dentry = dentry;
    
    inode->dir_cnt = 0;
    inode->dentrys = NULL;
    
    // allocate datablocks
    for(int i = 0; i < 6; i++){
        inode->block_p[i] = inode->ino * 6 + i;
    }

    return inode;
}

// write the file and directory structure to the disk
// the file and directory structure in the memory
// start from the root dentry
int newfs_sync_inode(struct newfs_inode *inode) {
    struct newfs_inode_d  inode_d;
    struct newfs_dentry*  dentry_cursor;
    struct newfs_dentry_d dentry_d;
    
    // assign for inode
    int ino = inode->ino;
    inode_d.ino = ino;
    inode_d.size = inode->size;
    inode_d.ftype = inode->dentry->ftype;
    inode_d.dir_cnt = inode->dir_cnt;
    for (int i = 0; i < 6; i++) {
        inode_d.block_p[i] = inode->block_p[i];
    }
    
    int offset;

    // write inode to the disk offset according to the ino
    if (newfs_driver_write(NEWFS_INO_OFS(ino), (uint8_t *)&inode_d, sizeof(struct newfs_inode_d)) != NEWFS_ERROR_NONE) {
        NEWFS_DBG("[%s] io error\n", __func__);
        return -NEWFS_ERROR_IO;
    }

    // if inode is directory file
    if (NEWFS_IS_DIR(inode)) {                          
        dentry_cursor = inode->dentrys;
        offset = NEWFS_DATA_OFS(inode->block_p[0]);

        // traverse each dentry
        while (dentry_cursor != NULL){
            
            // the content to write back
            memcpy(dentry_d.fname, dentry_cursor->fname, MAX_NAME_LEN);
            dentry_d.ftype = dentry_cursor->ftype;
            dentry_d.ino = dentry_cursor->ino;
                    
            // write back dentry information to disk
            if (newfs_driver_write(offset, (uint8_t *)&dentry_d, sizeof(struct newfs_dentry_d)) != NEWFS_ERROR_NONE){
                NEWFS_DBG("[%s] io error\n", __func__);
                return ~NEWFS_ERROR_IO;                     
            }
            
            // if the dentry is not empty, call function newfs_sync_inode() recursively
            if (dentry_cursor->inode != NULL) {
                newfs_sync_inode(dentry_cursor->inode);
            }
            
            // traverse its brother dentry
            dentry_cursor = dentry_cursor->brother;

            offset += sizeof(struct newfs_dentry_d);
        }
    }
    return 0;
}

// drop dentries based on inode in the memory
int newfs_drop_inode(struct newfs_inode * inode) {
    struct newfs_dentry* dentry_cursor;
    struct newfs_dentry* dentry_to_free;
    struct newfs_inode* inode_cursor;
    int byte_cursor, bit_cursor; 

    // the inode of the root directory can't be freed
    if (inode == newfs_super.root_dentry->inode) {
        return NEWFS_ERROR_INVAL;
    }

    // if inode is directory file
    if (NEWFS_IS_DIR(inode)) {
        dentry_cursor = inode->dentrys;
        
        // recursive drop
        while (dentry_cursor) {   
            inode_cursor = dentry_cursor->inode;
            newfs_drop_inode(inode_cursor);
            newfs_drop_dentry(inode, dentry_cursor);
            dentry_to_free = dentry_cursor;
            dentry_cursor = dentry_cursor->brother;
            free(dentry_to_free);
        }

        // in the inodeblock bitmap
        // set the released dentry’s position 0
        byte_cursor = inode->ino / UINT8_BITS;
        bit_cursor = inode->ino % UINT8_BITS;
        newfs_super.map_inode[byte_cursor] &= (uint8_t)(~(0x1 << bit_cursor));
    }

    // if inode is ordinary file
    else if (NEWFS_IS_REG(inode)) {
        // in the inodeblock bitmap
        // set the released dentry’s position 0
        byte_cursor = inode->ino / UINT8_BITS;
        bit_cursor = inode->ino % UINT8_BITS;
        newfs_super.map_inode[byte_cursor] &= (uint8_t)(~(0x1 << bit_cursor));
        char *buf = (char*) malloc(sizeof(NEWFS_IO_SZ()));
        memset(buf, 0, sizeof(buf));

        for (int i = 0; i < 6; i++){
            // in the datablock bitmap
            // set the released datablock’s position 0
            byte_cursor = inode->block_p[i] / UINT8_BITS;
            bit_cursor = inode->block_p[i] % UINT8_BITS;
            newfs_super.map_inode[byte_cursor] &= (uint8_t)(~(0x1 << bit_cursor));
            newfs_driver_write(NEWFS_DATA_OFS(inode->block_p[i]), (uint8_t *)buf, NEWFS_IO_SZ() * 2);
        }      
        free(inode);
    }
    return NEWFS_ERROR_NONE;
}

// search for [dir] entry of inode
struct newfs_dentry* newfs_get_dentry(struct newfs_inode * inode, int dir) {
    struct newfs_dentry* dentry_cursor = inode->dentrys;
    int cnt = 0;
    // traverse all the dentries
    while (dentry_cursor){
        if (dir == cnt) {
            return dentry_cursor;
        }
        cnt++;
        dentry_cursor = dentry_cursor->brother;
    }
    return NULL;
}

// read dentries in inodeblocks and datablocks
struct newfs_inode* newfs_read_inode(struct newfs_dentry * dentry, int ino) {
    struct newfs_inode* inode = (struct newfs_inode*)malloc(sizeof(struct newfs_inode));
    struct newfs_inode_d inode_d;
    struct newfs_dentry* sub_dentry;
    struct newfs_dentry_d dentry_d;
    int dir_cnt = 0;

    if (newfs_driver_read(NEWFS_INO_OFS(ino), (uint8_t *)&inode_d, sizeof(struct newfs_inode_d)) != NEWFS_ERROR_NONE) {
        NEWFS_DBG("[%s] io error\n", __func__);
        return NULL;                    
    }

    // read the content
    inode->dir_cnt = 0;
    inode->ino = inode_d.ino;
    inode->size = inode_d.size;
    inode->dentry = dentry;
    inode->dentrys = NULL;
    for (int i = 0; i < 6; i++){
        inode->block_p[i] = inode_d.block_p[i];
    }

    // if inode is directory file
    if (NEWFS_IS_DIR(inode)) {
        dir_cnt = inode_d.dir_cnt;

        // traverse each dentry
        for (int i = 0; i < dir_cnt; i++){
            if (newfs_driver_read(NEWFS_DATA_OFS(inode->block_p[0]) + i * sizeof(struct newfs_dentry_d), (uint8_t *)&dentry_d, 
                                sizeof(struct newfs_dentry_d)) != NEWFS_ERROR_NONE) {
                NEWFS_DBG("[%s] io error\n", __func__);
                return NULL;                    
            }
            sub_dentry = new_dentry(dentry_d.fname, dentry_d.ftype);
            sub_dentry->parent = inode->dentry;
            sub_dentry->ino    = dentry_d.ino; 
            
            // allocate for sub_dentry, point to inode
            newfs_alloc_dentry(inode, sub_dentry);
        }
        for (int i = 0; i < 6; i++){
            inode->block_p[i] = inode_d.block_p[i];
        } 
    }

    // if inode is ordinary file
    else if (NEWFS_IS_REG(inode)) {
        for (int i = 0; i < 6; i++){
            inode->block_p[i] = inode_d.block_p[i];
        }
    }
    return inode;
}

// look up the dentry, find file by path
struct newfs_dentry* newfs_lookup(const char * path, boolean* is_find, boolean* is_root) {
    struct newfs_dentry* dentry_cursor = newfs_super.root_dentry;
    struct newfs_dentry* dentry_ret = NULL;
    struct newfs_inode*  inode; 
    int   total_lvl = newfs_calc_lvl(path); // calculate the level of the path
    int   lvl = 0;
    boolean is_hit;
    char* fname = NULL;
    char* path_cpy = (char*)malloc(sizeof(path));
    *is_root = FALSE;
    strcpy(path_cpy, path);

    // root dentry
    if (total_lvl == 0) {
        *is_find = TRUE;
        *is_root = TRUE;
        dentry_ret = newfs_super.root_dentry;
    }

    fname = strtok(path_cpy, "/");      // split string     
    
    // traverse each level of dentry
    while (fname) {   
        lvl++;
        
        // read inode
        if (dentry_cursor->inode == NULL) {     
            newfs_read_inode(dentry_cursor, dentry_cursor->ino);
        }
        inode = dentry_cursor->inode;

        // error: it is a ordinary file but its level is less than total level
        if (NEWFS_IS_REG(inode) && lvl < total_lvl) {
            NEWFS_DBG("[%s] not a dir\n", __func__);
            dentry_ret = inode->dentry;
            break;
        }

        // if inode is directory file
        if (NEWFS_IS_DIR(inode)) {
            dentry_cursor = inode->dentrys;
            is_hit        = FALSE;

            // traverse each dentry
            while (dentry_cursor){
                
                // find file have the same name with fname
                if (memcmp(dentry_cursor->fname, fname, strlen(fname)) == 0) {
                    is_hit = TRUE;
                    break;
                }
                dentry_cursor = dentry_cursor->brother;
            }
            
            // error: can't find file
            if (!is_hit) {
                *is_find = FALSE;
                NEWFS_DBG("[%s] not found %s\n", __func__, fname);
                dentry_ret = inode->dentry;
                break;
            }

            // successfully find the file
            if (is_hit && lvl == total_lvl) {
                *is_find = TRUE;
                dentry_ret = dentry_cursor;
                break;
            }
        }
        fname = strtok(NULL, "/");  // end cycle
    }

    if (dentry_ret->inode == NULL) {
        dentry_ret->inode = newfs_read_inode(dentry_ret, dentry_ret->ino);
    }
    
    return dentry_ret;
}

// mount
int newfs_mount(struct custom_options options){
    struct newfs_super_d    newfs_super_d; 
    struct newfs_dentry*    root_dentry;
    struct newfs_inode*     root_inode;
    int                     file_num;
    int                     inode_blks; 
    int                     data_blks;
    int                     super_blks;
    int                     sz_bitmap_ino;
    int                     sz_bitmap_data;
    uint8_t*                inode_bitmap;
    uint8_t*                data_bitmap;
    boolean                 is_init = FALSE;

    newfs_super.is_mounted = FALSE;     // not mounted

    // read information from disk
    int driver_fd = ddriver_open(options.device);
    if (driver_fd < 0)      return driver_fd;   // fail to read
    newfs_super.fd = driver_fd;

    // the size of disk and io
    ddriver_ioctl(NEWFS_DRIVER(), IOC_REQ_DEVICE_SIZE, &newfs_super.sz_disk);
    ddriver_ioctl(NEWFS_DRIVER(), IOC_REQ_DEVICE_IO_SZ, &newfs_super.sz_io);

    // is read super_d successfully
    if (newfs_driver_read(NEWFS_SUPER_OFS, (uint8_t *)(&newfs_super_d), sizeof(struct newfs_super_d)) != NEWFS_ERROR_NONE) {
        return ~NEWFS_ERROR_IO;
    }   

    // if not already initialized
    if (newfs_super_d.magic_num != NEWFS_MAGIC_NUM) {     // magic_num = none 
        
        // calculate the size of each part
        super_blks = NEWFS_ROUND_UP(sizeof(struct newfs_super_d), NEWFS_IO_SZ()) / NEWFS_IO_SZ();
        file_num = NEWFS_DISK_SZ() / ((NEWFS_DATA_PER_FILE + NEWFS_INODE_PER_FILE) * NEWFS_IO_SZ());
        inode_blks = NEWFS_ROUND_UP(NEWFS_ROUND_UP(file_num, UINT32_BITS), NEWFS_IO_SZ()) / NEWFS_IO_SZ();
        data_blks = NEWFS_ROUND_UP(NEWFS_ROUND_UP(file_num * NEWFS_DATA_PER_FILE, UINT32_BITS), NEWFS_IO_SZ()) / NEWFS_IO_SZ();

        // calculate the layout 
        newfs_super_d.inode_blks = inode_blks;
        newfs_super_d.data_blks = data_blks;
        newfs_super.max_ino = (file_num - super_blks - inode_blks); 
        newfs_super_d.map_inode_offset = NEWFS_SUPER_OFS + NEWFS_BLKS_SZ(super_blks);
        newfs_super_d.map_data_offset = newfs_super.map_inode_offset + NEWFS_BLKS_SZ(inode_blks);;
        newfs_super_d.inode_offset = newfs_super.map_data_offset + NEWFS_BLKS_SZ(data_blks);
        newfs_super_d.data_offset = newfs_super_d.map_inode_offset + NEWFS_BLKS_SZ(file_num * NEWFS_DATA_PER_FILE);
        newfs_super_d.sz_usage = 0;

        is_init = TRUE;
    }

    // establish root dentry
    root_dentry = new_dentry("/", NEWFS_DIR);
    
    // assign for superblock in memory
    newfs_super.sz_usage = newfs_super_d.sz_usage;
    newfs_super.map_inode = (uint8_t *)malloc(NEWFS_BLKS_SZ(newfs_super_d.inode_blks));
    newfs_super.map_data = (uint8_t *)malloc(NEWFS_BLKS_SZ(newfs_super_d.data_blks));
    newfs_super.inode_offset = newfs_super_d.inode_offset;
    newfs_super.inode_blks = newfs_super_d.inode_blks;
    newfs_super.map_inode_offset = newfs_super_d.map_inode_offset;
    newfs_super.data_offset = newfs_super_d.data_offset;
    newfs_super.data_blks = newfs_super_d.data_blks;
    newfs_super.map_data_offset = newfs_super_d.map_data_offset;

    // read inodeblock bitmap
    if (newfs_driver_read(newfs_super_d.map_inode_offset, (uint8_t *)(newfs_super.map_inode), NEWFS_BLKS_SZ(inode_blks)) != 0){
        return ~NEWFS_ERROR_NONE;
    }

    // read datablock bitmap
    if (newfs_driver_read(newfs_super_d.map_data_offset, (uint8_t *)(newfs_super.map_data), NEWFS_BLKS_SZ(data_blks)) != 0){
        return ~NEWFS_ERROR_NONE;
    }

    // allocate root dentry
    if (is_init) {
        root_inode = newfs_alloc_inode(root_dentry);
        newfs_sync_inode(root_inode);
    }
    
    // read root dentry in disk
    root_inode = newfs_read_inode(root_dentry, NEWFS_ROOT_INO);
    root_dentry->inode = root_inode;
    newfs_super.root_dentry = root_dentry;
    newfs_super.is_mounted  = TRUE;

    // debug
    newfs_dump_map();

    return NEWFS_ERROR_NONE;
}

int newfs_umount() {
    struct newfs_super_d  newfs_super_d; 
    
    // ensure it is mounted
    if (!newfs_super.is_mounted)    return NEWFS_ERROR_NONE;

    // synchronize file information in memory
    newfs_sync_inode(newfs_super.root_dentry->inode); 
                                                    
    newfs_super_d.magic_num = NEWFS_MAGIC_NUM;
    newfs_super_d.max_ino = newfs_super.max_ino;
    newfs_super_d.inode_offset = newfs_super.inode_offset;
    newfs_super_d.inode_blks = newfs_super.inode_blks;
    newfs_super_d.map_inode_offset = newfs_super.map_inode_offset;
    newfs_super_d.data_offset = newfs_super.data_offset;
    newfs_super_d.data_blks = newfs_super.data_blks;
    newfs_super_d.map_data_offset = newfs_super.map_data_offset;
    newfs_super_d.sz_usage = newfs_super.sz_usage;

    // write superblock
    if (newfs_driver_write(NEWFS_SUPER_OFS, (uint8_t *)&newfs_super_d, sizeof(struct newfs_super_d)) != NEWFS_ERROR_NONE) {
        return -NEWFS_ERROR_IO;
    }
    
    // write inodeblock bitmap
    if (newfs_driver_write(newfs_super_d.map_inode_offset, (uint8_t *)(newfs_super.map_inode), NEWFS_BLKS_SZ(newfs_super_d.inode_blks)) != NEWFS_ERROR_NONE) {
        return -NEWFS_ERROR_IO;
    }

    // write datablock bitmap
    if (newfs_driver_write(newfs_super_d.map_data_offset, (uint8_t *)(newfs_super.map_data), NEWFS_BLKS_SZ(newfs_super_d.data_blks)) != NEWFS_ERROR_NONE) {
        return -NEWFS_ERROR_IO;
    }

    free(newfs_super.map_inode);
    free(newfs_super.map_data);
    ddriver_close(NEWFS_DRIVER());

    return NEWFS_ERROR_NONE;
}