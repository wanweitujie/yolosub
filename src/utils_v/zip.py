import zipfile
import os
import time
import logging
from typing import Dict, Optional

 # 格式化大小显示函数
def format_size(size_bytes):
    """格式化字节大小为可读格式"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes/(1024*1024):.2f} MB"
    else:
        return f"{size_bytes/(1024*1024*1024):.2f} GB"

def zip_with_max_compression(source_path, output_zip, extra_contents: Optional[Dict[str, str]] = None):
    """
    创建最高压缩率的ZIP文件。
    
    可以处理两种情况：
    1. source_path 是文件：直接压缩该文件
    2. source_path 是目录：压缩整个目录(包括空文件夹)
    
    Args:
        source_path: 源文件或目录的路径
        output_zip: 输出的ZIP文件路径
        extra_contents: 可选。{归档内路径: 文件内容字符串}，在主内容写完后追加写入。
            None（默认）时行为与不传该参数完全一致。内容以 str 直接写入，按 UTF-8 编码。

    Returns:
        output_zip: 创建的ZIP文件路径
    """
    # 验证源路径存在
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"源路径不存在: {source_path}")
    
    start_time = time.time()
    
    # 统计变量
    total_files = 0
    total_folders = 0
    original_size = 0
    
    added_dirs = set()
    
    with zipfile.ZipFile(output_zip, 'w', 
                         compression=zipfile.ZIP_DEFLATED, 
                         compresslevel=9) as zipf:
        
        # 情况1: 源路径是文件
        if os.path.isfile(source_path):
            # 获取文件名作为归档名
            # arcname = os.path.basename(source_path)            
            ext = os.path.splitext(source_path)[1]
            arcname = "model" + ext 
            # 添加到ZIP
            zipf.write(source_path, arcname)
            # 更新统计
            total_files = 1
            original_size = os.path.getsize(source_path)
            
        # 情况2：目录 → 压缩其内部所有内容（不包含目录名本身）
        else:
            parent_dir = os.path.abspath(source_path)
            
            # 遍历源目录
            for root, dirs, files in os.walk(source_path):
                # 计算当前目录的相对路径
                rel_root = os.path.relpath(root, parent_dir)
                
                # 添加当前目录（如果不是根目录）
                if rel_root != '.':
                    dir_path = rel_root + '/' if not rel_root.endswith('/') else rel_root
                    if dir_path not in added_dirs:
                        zipf.writestr(dir_path, '')
                        total_folders += 1
                        added_dirs.add(dir_path)
                
                # 添加当前目录下的所有子目录 - 这里正确使用dirs列表
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    rel_dir_path = os.path.relpath(dir_path, parent_dir)
                    # 确保目录路径以斜杠结尾
                    if not rel_dir_path.endswith('/'):
                        rel_dir_path += '/'
                        
                    if rel_dir_path not in added_dirs:
                        zipf.writestr(rel_dir_path, '')
                        total_folders += 1
                        added_dirs.add(rel_dir_path)
                
                # 添加文件
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    original_size += file_size
                    
                    file_rel_path = os.path.relpath(file_path, parent_dir)
                    zipf.write(file_path, file_rel_path)
                    total_files += 1

        # 追加内存中的附加条目(如 classes.txt):内容直接写入,无需落盘临时文件。
        # 不计入上方 total_files / original_size 统计(纯日志层面,不影响归档内容)。
        for arcname, content in (extra_contents or {}).items():
            if arcname in zipf.namelist():
                logging.getLogger(__name__).warning(
                    f"extra_contents 条目 {arcname} 与既有条目重名,已跳过"
                )
                continue
            zipf.writestr(arcname, content)  # str 默认 UTF-8 编码

    # 计算压缩后大小和执行时间
    compressed_size = os.path.getsize(output_zip)
    elapsed_time = time.time() - start_time
    
   
    
    # 计算压缩率
    if original_size > 0:
        compression_ratio = original_size / compressed_size
        reduction_percent = (1 - compressed_size/original_size) * 100
    else:
        compression_ratio = 0
        reduction_percent = 0
    
    # 打印统计信息
    print(f"\nZIP文件创建成功: {output_zip}")
    print(f"处理耗时: {elapsed_time:.2f} 秒")
    print(f"{'统计信息':-^40}")
    print(f"总条目数: {total_files + total_folders}")
    
    if os.path.isdir(source_path):
        print(f"文件数: {total_files}")
        print(f"文件夹数: {total_folders}")
    
    print(f"原始大小: {format_size(original_size)}")
    print(f"压缩后大小: {format_size(compressed_size)}")
    
    if compression_ratio > 0:
        print(f"压缩率: {compression_ratio:.2f}x ({reduction_percent:.1f}% 减少)")
    
    return output_zip