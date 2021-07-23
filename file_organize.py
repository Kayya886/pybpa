# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: file_organize
# Author:       MilkyDesk
# Date:         2021/7/7 23:40
# Description:
#   
"""

# ------------------------- params --------------------------


# ------------------------- params --------------------------
import re

from dat.bpa_dat import DAT
from swi.bpa_swi import SWI


class JyLkOrganizer:
    """
    济宇的文件组织方式
    liukai的文件组织方式：
    _path全部以'/'结尾
    """
    def __init__(self, case_system, root_path, source_path, target_path, dat_in_root, swi_in_root):
        """

        @param case_system:
        @param root_path:
        @param source_path:
        @param target_path:
        @param dat_in_root:
        @param swi_in_root:
        """
        self.case_system = str(case_system)
        self.root_path = root_path + self.case_system + '/'
        self.source_path = source_path
        self.target_path = target_path
        self.dat_in_root = dat_in_root
        self.swi_in_root = swi_in_root
        self.dat = None
        self.get_dat()
        self.swi = None
        self.get_swi()

    def get_dat(self, operation_folder: str = None):
        if self.dat_in_root or not operation_folder:
            if self.dat is None:
                self.dat = DAT.build_from_folder(self.source_path + '0_100_0/')
        else:
            self.dat = DAT.build_from_folder(self.source_path + operation_folder + '/')
        return self.dat

    def get_swi(self, operation_folder: str = None):
        if self.swi_in_root or not operation_folder:
            if self.swi is None:
                self.swi = SWI.build_from_folder(self.source_path)
        else:
            self.swi = SWI.build_from_folder(self.source_path + operation_folder + '/')
            self.swi.dat = self.get_dat()
        return self.swi

    def parse_operation_name(self, operation_name: str):
        operation_info = re.sub('_', ' ', operation_name).split()

        assert len(operation_info) >= 3
        pre_cut_line_num = int(operation_info[0])
        load_level = int(operation_info[1])
        pre_cut_line_list = [int(v) for v in operation_info[2:(2 + pre_cut_line_num)]]
        return pre_cut_line_num, load_level, pre_cut_line_list

    def parse_swx_name(self, swx_name: str):
        """
        解析swx文件的名称，得到故障切除线路和故障节点
        默认结构：a(b-c).swx
        """
        if 'bus' in swx_name:
            return self._parse_swx_name_old(swx_name)
        post_cut_line_index, fault_bus_index = None, None
        try:  # 有些名称不规则所以要try
            swx_info = re.sub('[.SWX()-]', ' ', swx_name.upper()).split()
            post_cut_line_index = int(swx_info[1])  # 默认这里就是dat读取到的line的原始顺序
            fault_bus_index = int(swx_info[2])  # 默认这里就是dat读取到的bus的原始顺序
        except:
            pass
        return post_cut_line_index, fault_bus_index

    def _parse_swx_name_old(self, swx_name: str):
        """结构：a(busb-busc(b)).swx, 只会出现在case39的0_"""
        swx_info = re.sub('[()-]', ' ', swx_name[:-4]).split()

        post_cut_line_index = -1
        for i, l in enumerate(self.dat.branch_name):
            if swx_info[1].ljust(8) in l and swx_info[2].ljust(8) in l:
                if post_cut_line_index == -1:
                    post_cut_line_index = i
                else:
                    raise ValueError('解析方式有误,出现了多条线路包含swx_info[1+2]！')
        if swx_info[1] == 'bus' + swx_info[3]:
            fault_bus_index = self.dat.bus_order[swx_info[1].ljust(8) + '100 ']
        else:
            fault_bus_index = self.dat.bus_order[swx_info[2].ljust(8) + '100 ']

        assert self.dat.bus_name[fault_bus_index] in self.dat.branch_name[post_cut_line_index]
        return post_cut_line_index, fault_bus_index