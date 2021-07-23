# -*- coding:utf-8 -*-

# ===========================================================
# Name:         transient_stability: gen_dat
# Author:       MilkyDesk
# Date:         2021/3/11 10:21
# Description:
#   生成方式文件夹，每个文件夹包含dat文件
#   然后手动打开南瑞程序生成
# ===========================================================
import os
import re
from multiprocessing import Pool

from tqdm import tqdm

import bpa_utils

# ------------------------- fields --------------------------
# path_dats = 'C:/Users/MilkyDesk/Downloads/' + 'dats/'
case_system = 300

path_root = 'E:/Data/transient_stability/' + str(case_system) + '/'
"""唯一一个与节点数量有关的路径"""

# path_dats = path_root + 'dats/'
path_dats = 'F:/dats/'
"""方式文件夹的存放路径"""

path_bpa = path_root + 'bpa/'
path_pretreated = path_root + 'pretreated/'

dat_file = path_bpa + '0_100_0/0_100_0.dat'

ignored_suffix_in_bpa_folder = ['.db', '.plog', '.prj', '.txt', '.MRK', '.dat', '.error']

bpa_exe_path = 'D:/OneDrive/桌面/PsdEdit/pfnt.exe'

bus_list, line_list, bus_text, line_text, _, _, _, _ = utils.read_datNF(dat_file)
bus_dict = {name[1]: i for i, name in enumerate(bus_list)}


bus_line_start_flag = ['B', 'X']
branch_line_start_flag = ['L', 'T', 'E', 'R']
# ------------------------- fields --------------------------
"""
1. 遍历生成dat和文件夹
2. 提供一个接口，可以根据运行方式信息使用
"""


def save_lines_to_file(dat_lines, target_path, file_name):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    file = open(target_path + file_name, 'w')
    for lines in dat_lines:
        file.write(lines)
    file.close()


def check_pre_fault_dat(pre_fault_dat_path, pre_cut_line_list):
    """
    检查故障前dat文件并返回被修正后的lines
    @param pre_fault_dat_path:
    @param pre_cut_line_list:
    @return: 处理后的dat_lines
    """
    pre_fault_dat_lines = open(pre_fault_dat_path, 'r', errors='ignore').readlines()
    branch_line_start_index = 100000
    branch_line_end_index = 100000
    for i, line in enumerate(pre_fault_dat_lines):
        if i < branch_line_start_index and line[0] in branch_line_start_flag:
            branch_line_start_index = i
        if branch_line_start_index < i < branch_line_end_index and pre_fault_dat_lines[i][0] not in branch_line_start_flag:
            branch_line_end_index = i

    # 将线路数据处理成：只有pre_cut_line被切除，其他线路没有被切除的状态
    dat_lines = []
    # 1. 线路前部分原样保留
    for line in pre_fault_dat_lines[:branch_line_start_index]:
        dat_lines.append(line)
    # 2. 线路部分切除预定线路
    for i, line in enumerate(pre_fault_dat_lines[branch_line_start_index:branch_line_end_index]):
        if i in pre_cut_line_list:
            dat_lines.append(commented(line))
        else:
            dat_lines.append(uncommented(line))
    # 3. 线路后部分原样保留
    for line in pre_fault_dat_lines[branch_line_end_index:]:
        dat_lines.append(line)
    return dat_lines


def generate_post_dat(pre_fault_dat_lines, fault_bus, post_cut_line):
    """
    在给定的dat文件基础上，
    按照指定的方式修改文件
    得到故障中（近似）和故障后的dat文件，并分别跑潮流得到pfo文件
    @param pre_fault_dat_path: 故障前dat文件，包含基本潮流外，还有负荷调整信息，可能包含错误的断线信息
    @param fault_bus: 母线出口处短路（线路出口短路）的母线编号（0开始）
    @param post_cut_line: 故障后切除的线路编号（0开始）
    @return: 故障时的dat文件完整路径，故障后的dat文件完整路径
    """

    branch_line_start_index = 100000
    for i, line in enumerate(pre_fault_dat_lines):
        if i < branch_line_start_index and line[0] in branch_line_start_flag:
            branch_line_start_index = i

    post_fault_dat_lines = []

    for i in range(pre_fault_dat_lines.__len__()):
        line = pre_fault_dat_lines[i]
        # 去除节点电压控制
        if line[0:2] == 'BV':
            line = 'B ' + line[2:30] + '\n'
        elif line[0:2] == 'BQ':
            line = line[:47] + '99990-9999' + line[57:]
        # 切除目标线路
        if i == branch_line_start_index + post_cut_line:
            post_fault_dat_lines.append(commented(line))
        else:
            post_fault_dat_lines.append(line)

    return post_fault_dat_lines


def build_case_multiprocess(operation_name):
    n_case = 0
    n_operation = 0
    if all(suffix not in operation_name for suffix in ignored_suffix_in_bpa_folder):
        # 计数器
        n_operation = 1
        # 遍历方式文件夹
        operation_path = path_bpa + operation_name
        operation_info = re.sub('_', ' ', operation_name).split()
        if operation_info[0] == '0':
            pre_cut_line_list = []
        elif operation_info[0] == '1':
            pre_cut_line_list = [int(operation_info[2])]
        elif operation_info[0] == '2':
            pre_cut_line_list = [int(operation_info[2]), int(operation_info[3])]
        else:
            print("is_debug: operation_name: ", operation_name)
            pre_cut_line_list = []

        pre_fault_dat_path = operation_path + '/' + operation_name + '.dat'
        pre_fault_dat_lines = check_pre_fault_dat(pre_fault_dat_path, pre_cut_line_list)
        save_lines_to_file(pre_fault_dat_lines, path_dats + operation_name + '/', operation_name + '.dat')

        for file in os.listdir(operation_path):
            if '.SWX' in file:  # x((x)-(x)).swi
                swx_file = re.sub('[.SWX()-]', ' ', file)  # x(#pre cut) x(#fault bline) x(#fault Bus)

                # 1. 产生新的dat
                # swx有些命名是 bus1--bus2 有些是 fault_line - fault_bus
                swx_info = swx_file.split()
                if len(swx_info) < 3:  # 有些swx直接不要了
                    continue
                # swx有两种命名格式
                if 'Bus' in swx_info[1]:  # 第一种： k((busa——busb)(x)), k:n-k, x:morning/b短路端
                    if swx_info[3] in swx_info[1]:
                        fault_bus = bus_dict.get(swx_info[1])
                    else:
                        fault_bus = bus_dict.get(swx_info[2])

                    bus_a = bus_text[bus_dict.get(swx_info[1])][1]  # like 'bus1100.0'
                    bus_b = bus_text[bus_dict.get(swx_info[2])][1]
                    for i, branch in enumerate(line_text):
                        if (branch[1] == bus_a and branch[2] == bus_b) or (branch[2] == bus_a and branch[1] == bus_b):
                            post_cut_line = i
                            break
                else:  # 第二种： k(l-b), k:n-k, l:fault_line, b:fault_bus
                    post_cut_line = int(swx_info[1])
                    fault_bus = int(swx_info[2])

                post_fault_dat_lines = generate_post_dat(pre_fault_dat_lines,
                                                         fault_bus,
                                                         post_cut_line)
                # 2. 执行bpa,读pfo
                operation_name_post = operation_name[:-1] + '2_' + str(post_cut_line)
                save_lines_to_file(post_fault_dat_lines, path_dats + operation_name_post + '/',
                                   operation_name_post + '.dat')

                # 3. 检查生成的文件是不是太小
                file_too_small = ''
                if os.path.getsize(path_dats + operation_name + '/' + operation_name + '.dat') < 4859:
                    file_too_small += operation_name + ' '
                if os.path.getsize(path_dats + operation_name_post + '/' + operation_name_post + '.dat') < 4859:
                    file_too_small += operation_name_post + ' '

                n_case += 1

    return [n_operation, n_case, file_too_small]


def prepare_data():
    listdir = os.listdir(path_bpa)
    listdir = [x for x in listdir if '.' not in x]
    count = [0, 0]
    file_too_small = []
    # 单线程
    # for operation_name in tqdm(listdir, desc='gen_dats'):
    #     result = build_case_multiprocess(operation_name)
    #     killed_count[0] += result[0]
    #     killed_count[1] += result[1]

    # 多线程1
    # pool = ThreadPool()
    # pool.map(build_case_multiprocess, listdir)
    # pool.close()

    # 多进程2
    with Pool(processes=6) as p:
        with tqdm(total=len(listdir)) as pbar:
            for i, result in enumerate(p.imap_unordered(build_case_multiprocess, listdir)):
                pbar.update()
                count[0] += result[0]
                count[1] += result[1]
                # if result[2] != '':
                #     file_too_small.append(result[2])

    print('operation: ', count[0], '\tcase: ', count[1])  # , '\ttoo_small: ', file_too_small.__len__())
    # for x in file_too_small:
    #     print(x)
    # 多进程3
    # p_tqdm.p_map(build_case_multiprocess, listdir,
    #              # num_cpus=1
    #              )


def commented(s):
    """将一行注释掉"""
    return '.' + s


def uncommented(s):
    """将一行反注释"""
    if s:
        i = 0
        while i < len(s) and s[i] == '.':
            i += 1
        return s[i:]
    return s


if __name__ == '__main__':

    if not os.path.exists(path_dats):
        os.makedirs(path_dats)
    prepare_data()
