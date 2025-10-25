
import os

def count_files_in_subfolders(root_dir, prefix=""):
	for subdir in sorted(os.listdir(root_dir)):
		subpath = os.path.join(root_dir, subdir)
		if os.path.isdir(subpath):
			file_count = 0
			for root, dirs, files in os.walk(subpath):
				file_count += len([f for f in files if os.path.isfile(os.path.join(root, f))])
			print(f"{prefix}{subdir}: {file_count} files")
			# 递归打印下一级
			count_files_in_subfolders(subpath, prefix=prefix+"  ")

if __name__ == '__main__':
	# import argparse
	# parser = argparse.ArgumentParser(description='统计每个子文件夹下的文件数')
	# parser.add_argument('root_dir', type=str, help='根目录路径')
	# args = parser.parse_args()
	# count_files_in_subfolders(args.root_dir)
	# root_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/3_contact_task'
	root_dir = '/home/yifei/code/Med_CV/MedMamba/dataset/4_spatial_task'
	count_files_in_subfolders(root_dir)


