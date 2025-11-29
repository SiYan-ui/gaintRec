#!/usr/bin/env python3
"""
将 data/GaitDatasetB-silh/ 中的 001.tar.gz 到 124.tar.gz 全部解压到同一目录下（保留各自包内原有目录结构）。

用法示例：
	python src/utility/unzip_dataset.py --data-dir data/GaitDatasetB-silh

脚本要点：
- 安全解压（防止路径遍历）
- 跳过已解压的包（如果发现顶级目录已存在）
- 打印处理进度与错误信息
"""

import argparse
from pathlib import Path
import tarfile
import os
import sys
from tqdm import tqdm


def is_within_directory(directory: str, target: str) -> bool:
	# 判断 target 路径是否在 directory 内，防止路径遍历攻击
	abs_directory = os.path.abspath(directory)
	abs_target = os.path.abspath(target)
	try:
		# commonpath计算最长公共路径
		return os.path.commonpath([abs_directory]) == os.path.commonpath([abs_directory, abs_target])
	except Exception:
		return False


def safe_extract(tar: tarfile.TarFile, path: str = "."):
	"""安全提取 tar，防止路径遍历漏洞"""
	for member in tar.getmembers():
		member_path = os.path.join(path, member.name)
		if not is_within_directory(path, member_path):
			raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")
	tar.extractall(path)


def top_level_dir_of_tar(tar: tarfile.TarFile):
	"""返回 tar 中第一个成员的顶级路径（用于判断是否已解压）"""
	members = [m for m in tar.getmembers() if m.name and m.name.strip()]
	if not members:
		return None
	first = members[0].name
	top = first.split('/')[0]
	return top


def extract_all_archives(data_dir: Path, pattern: str = "*.tar.gz", skip_existing: bool = True):
	files = sorted(list(data_dir.glob(pattern)))    # 按照模板匹配文件
	if not files:
		print(f"No archives found in {data_dir} with pattern {pattern}")
		return

	for archive in tqdm(files, desc="Extracting archives"):
		try:
			if not archive.is_file():     # 跳过非文件（如目录）
				continue

			with tarfile.open(archive, 'r:gz') as tar:
				top = top_level_dir_of_tar(tar)
				if top:
					expected_dir = data_dir / top   # 拼接解压目录路径
					if skip_existing and expected_dir.exists():
						tqdm.write(f"Skipping {archive.name}: target {expected_dir} already exists")
						continue
				# perform safe extract
				safe_extract(tar, path=str(data_dir))
				tqdm.write(f"Extracted {archive.name}")
		except Exception as e:
			tqdm.write(f"Failed to extract {archive.name}: {e}")


def main():
	parser = argparse.ArgumentParser(description='Unpack multiple .tar.gz files in a directory')
	parser.add_argument('--data-dir', type=str, default='data/GaitDatasetB-silh', help='Directory containing the tar.gz files')
	parser.add_argument('--pattern', type=str, default='*.tar.gz', help='Glob pattern to match archives')
	parser.add_argument('--no-skip', action='store_true', help='Do not skip archives whose top-level dir already exists')
	args = parser.parse_args()

	data_dir = Path(args.data_dir)
	if not data_dir.exists() or not data_dir.is_dir():
		print(f"Data directory does not exist: {data_dir}")
		sys.exit(1)

	extract_all_archives(data_dir, pattern=args.pattern, skip_existing=not args.no_skip)


if __name__ == '__main__':
	main()
