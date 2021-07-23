# -*- coding: utf-8 -*-#
import os
import re
import sys
import time

import numpy as np
import subprocess
# from environment.environment import *

# -----------------------------------------------------------
# Name:         utils
# Author:       Jerem
# Date:         2021/2/11
# Description:  
# -----------------------------------------------------------

# ------------------------- fields --------------------------


def run_pf(dat_path):
    """
    谨慎使用。方法有问题，pfo为空。并且运行出错的概率反比于sleep时间。
    @param dat_path:
    @return:
    """
    # 使用call而非Popen：执行完这句再执行下一行
    # try:
    # format = open(path_pretreated2 + 'temp_out.txt', 'w')
    # e = open(path_pretreated2 + 'temp_err.txt', 'w')
    iter_times = subprocess.Popen(bpa_exe_path + ' ' + dat_path)
    # iter_times.kill()
    #     iter_times.wait()
    # finally:
    #     iter_times.kill()
    # 获取输出
    # out = sys.stdout.read()
    # err = sys.stderr.read()
    return iter_times


# 测试
if __name__ == '__main__':
    print('测试读取pfo')
    bus_list, line_list, bus_text, line_text, _, _, _, _ = read_datNF(dat_file)
    bus_dict = {name[1]: i for i, name in enumerate(bus_list)}
    read_pfo(r'D:\OneDrive\桌面\PsdEdit\0_100_0\0_100_0.dat', bus_dict)
