# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: gen_lsd
# Author:       MilkyDesk
# Date:         2021/7/23 11:12
# Description:
#   根据指定dat生成lsd的方法
"""

from dat.bpa_dat import DAT
# ------------------------- params --------------------------

# 目标dat所在的文件夹
dat_path = r'E:\Data\transient_stability\300\bpa\0_100_0'

# 形成文件夹结构的目录
target_path = ''

# 搜索范围
max_k = 2

# 还需要定义: 允许搜索的函数，将列表写入bpa和lsd的方法

# ------------------------- params --------------------------

# 预先生成的部分
base_dat = DAT.build_from_folder(dat_path)

load_levels = []
"""负荷水平"""

ls_branch = None
"""允许切除的线路集合"""

def gen_ls_branch():
    """根据base_dat产生可以切除的线路集合"""
    return []


def build_folder(cut_list, ls_list):

    """todo
    1. 新建文件夹
    2. 根据cutlist和loadlevel新建dat
    3. 根据

    """
    # 根据是否切线来搜索
    if len(cut_list) < max_k:
        for i, b in enumerate(ls_branch):
            if ls_list[i] and allow_cut_lines(cut_list + [i]):
                build_folder(cut_list + [i])
            else:
                ls_list[i] = False

    for load_level in load_levels:
        # 1， 2
        pass



def allow_cut_lines(line_list) -> bool:
    """base_dat切除line_list后检查连通性"""
    return True



# 执行脚本
if __name__ == '__main__':
    load_levels = []
    """负荷水平"""

    ls_branch = gen_ls_branch()
    ls_list = [True] * len(ls_branch)

    for load_level in load_levels:
        build_folder(load_level, cut_list=[], ls_list=ls_list)


