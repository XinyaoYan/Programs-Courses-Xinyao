#include "../include/newfs.h"
#include "../include/types.h"

/******************************************************************************
* SECTION: 宏定义
*******************************************************************************/
#define OPTION(t, p) {t, offsetof(struct custom_options, p), 1}

/******************************************************************************
* SECTION: 全局变量
*******************************************************************************/
static const struct fuse_opt option_spec[] = {/* 用于FUSE文件系统解析参数 */
											  OPTION("--device=%s", device),
											  FUSE_OPT_END};

struct custom_options newfs_options; /* 全局选项 */
struct newfs_super super;

/******************************************************************************
* SECTION: FUSE操作定义
*******************************************************************************/
static struct fuse_operations operations = {
	.init = newfs_init,		  /* mount文件系统 */
	.destroy = newfs_destroy, /* umount文件系统 */
	.mkdir = newfs_mkdir,	  /* 建目录，mkdir */
	.getattr = newfs_getattr, /* 获取文件属性，类似stat，必须完成 */
	.readdir = newfs_readdir, /* 填充dentrys */
	.mknod = newfs_mknod,	  /* 创建文件，touch相关 */
	.write = NULL,			  /* 写入文件 */
	.read = NULL,			  /* 读文件 */
	.utimens = newfs_utimens, /* 修改时间，忽略，避免touch报错 */
	.truncate = NULL,		  /* 改变文件大小 */
	.unlink = NULL,			  /* 删除文件 */
	.rmdir = NULL,			  /* 删除目录， rm -r */
	.rename = NULL,			  /* 重命名，mv */
	.open = NULL,
	.opendir = NULL,
	.access = NULL};

/******************************************************************************
* SECTION: 必做函数实现
*******************************************************************************/
// mount filesystem
void *newfs_init(struct fuse_conn_info *conn_info){
	if (newfs_mount(newfs_options) != NEWFS_ERROR_NONE) {
        NEWFS_DBG("[%s] mount error\n", __func__);
		fuse_exit(fuse_get_context()->fuse);
		return NULL;
	} 
	return NULL;
}

// umount filesystem
void newfs_destroy(void *p){
	if (newfs_umount() != NEWFS_ERROR_NONE) {
		NEWFS_DBG("[%s] unmount error\n", __func__);
		fuse_exit(fuse_get_context()->fuse);
		return;
	}
	return;
}

// make directory
int newfs_mkdir(const char *path, mode_t mode){
	(void)mode;
	boolean is_find, is_root;
	char* fname;
	
	// check: if the directory already exist, return error
	struct newfs_dentry* last_dentry = newfs_lookup(path, &is_find, &is_root);
	if (is_find)							return -NEWFS_ERROR_EXISTS;		
	if (NEWFS_IS_REG(last_dentry->inode))	return -NEWFS_ERROR_UNSUPPORTED;
	
	struct newfs_dentry* dentry;
	struct newfs_inode* inode;

	fname = newfs_get_fname(path);
	dentry = new_dentry(fname, NEWFS_DIR); 
	dentry->parent = last_dentry;

	// allocate inode
	inode = newfs_alloc_inode(dentry);

	// allocate dentry
	newfs_alloc_dentry(last_dentry->inode, dentry);
	
	return NEWFS_ERROR_NONE;
}

// gets the properties of a file or directory
int newfs_getattr(const char *path, struct stat *newfs_stat){
	boolean	is_find, is_root;

	// check: if the directory already exist, return error
	struct newfs_dentry* dentry = newfs_lookup(path, &is_find, &is_root);
	if (is_find == FALSE) 	return -NEWFS_ERROR_NOTFOUND;

	// if it is directory file
	if (NEWFS_IS_DIR(dentry->inode)) {
		newfs_stat->st_mode = S_IFDIR | NEWFS_DEFAULT_PERM;
		newfs_stat->st_size = dentry->inode->dir_cnt * sizeof(struct newfs_dentry_d);
	}

	// if it is ordinary file
	else if (NEWFS_IS_REG(dentry->inode)) {
		newfs_stat->st_mode = S_IFREG | NEWFS_DEFAULT_PERM;
		newfs_stat->st_size = dentry->inode->size;
	}

	// write properties
	newfs_stat->st_nlink = 1;
	newfs_stat->st_uid = getuid();
	newfs_stat->st_gid = getgid();
	newfs_stat->st_atime = time(NULL);
	newfs_stat->st_mtime = time(NULL);
	newfs_stat->st_blksize = NEWFS_IO_SZ();

	if (is_root) {
		newfs_stat->st_size	= newfs_super.sz_usage; 
		newfs_stat->st_blocks = NEWFS_DISK_SZ() / NEWFS_IO_SZ();
		newfs_stat->st_nlink = 2;		// root dentry's link is 2
	}
	return NEWFS_ERROR_NONE;
}

// traverse entries, fill their filenames in buf, and give them to fuse for output
int newfs_readdir(const char *path, void *buf, fuse_fill_dir_t filler, off_t offset, struct fuse_file_info *fi){
	// * path: the path relative to the mount point
	// * buffer: output buffer
	// * filler: * typedef int (*fuse_fill_dir_t) (void *buf, const char *name, const struct stat *stbuf, off_t off)
	// * * buf: name will be copied to buf
	// * * name: dentry's name
	// * * off: where the next offset will start at
	// * offset: which directory item?
	boolean	is_find, is_root;
	int		cur_dir = offset;

	struct newfs_dentry* dentry = newfs_lookup(path, &is_find, &is_root);
	struct newfs_dentry* sub_dentry;
	struct newfs_inode* inode;
	if (is_find) {
		inode = dentry->inode;
		sub_dentry = newfs_get_dentry(inode, cur_dir);
		if (sub_dentry) {
			// fill filenames to buf
			filler(buf, sub_dentry->fname, NULL, ++offset);
		}
		return NEWFS_ERROR_NONE;
	}
	return -NEWFS_ERROR_NOTFOUND;
}

// make new file
int newfs_mknod(const char *path, mode_t mode, dev_t dev){
	boolean	is_find, is_root;
	
	// check: if the directory already exist, return error
	struct newfs_dentry* last_dentry = newfs_lookup(path, &is_find, &is_root);
	if (is_find == TRUE) 	return -NEWFS_ERROR_EXISTS;

	struct newfs_dentry* dentry;
	struct newfs_inode* inode;
	char* fname;
	
	fname = newfs_get_fname(path);
	
	if (S_ISREG(mode)) 			dentry = new_dentry(fname, NEWFS_REG_FILE);
	else if (S_ISDIR(mode)) 	dentry = new_dentry(fname, NEWFS_DIR);

	// allocate inode
	dentry->parent = last_dentry;
	inode = newfs_alloc_inode(dentry);

	// allocate dentry
	newfs_alloc_dentry(last_dentry->inode, dentry);

	return NEWFS_ERROR_NONE;
}

// modify the time to prevent touch from reporting errors 
int newfs_utimens(const char *path, const struct timespec tv[2]){
	(void)path;
	return NEWFS_ERROR_NONE;
}

/******************************************************************************
* SECTION: FUSE入口
*******************************************************************************/
int main(int argc, char **argv){
	int ret;
	struct fuse_args args = FUSE_ARGS_INIT(argc, argv);

	newfs_options.device = strdup("/home/guests/190111014/ddriver");

	if (fuse_opt_parse(&args, &newfs_options, option_spec, NULL) == -1)
		return -1;

	ret = fuse_main(args.argc, args.argv, &operations, NULL);
	fuse_opt_free_args(&args);
	return ret;
}