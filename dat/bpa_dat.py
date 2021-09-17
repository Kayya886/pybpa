# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_dat
# Author:       MilkyDesk
# Date:         2021/6/29 9:59
# Description:
#   
"""
import inspect
import os

from typing import List
import numpy as np
import pandas as pd

from base import bpa_annotation
from base.bpa_uid import *
from base.bpa_base import _to_card
from base.bpa_str import bpa_card_line
from base.bpa_uid import _OneNameUid

from dat import bpa_dat_before_data, bpa_dat_data
"""
dat结构：
一级语句
网络数据前的控制语句
网络数据
控制语句
一级语句
"""



class DAT:
    """
    定位：
    提供Swi_Card的管理工具
    行为：
    1. 初始化: 读取文件
    2. __str__: 反向文本化功能。应该与_Card的组织类似。
    """
    annotation_card = {'.', ' ', 'C'}
    symmeyty_line_card_set = {'L ', 'LD', 'LM', 'T ', 'TP', 'R ', 'RZ', 'RV', 'RQ', 'RP', 'RN', 'RM'}
    asymmeyty_line_card_set = {'L+', 'E '}
    line_card_set = {'L ', 'LD', 'LM', 'T ', 'TP', 'R ', 'RZ', 'RV', 'RQ', 'RP', 'RN', 'RM', 'L+', 'E '}
    bus_card_set = {'B ', 'BV', 'BM', 'BQ', 'BS', 'BT', 'BE', 'BF', 'BJ', 'BK', 'BL', 'BX', 'BD', '+', 'X '}

    def __init__(self, *p):
        self.cards, self.can, self.cc1, self.cdt, self.cc2 = p
        self.bus, self.bus_indexs, self.bus_order, self.bus_name, \
        self.branch, self.branch_indexs, self.branch_order, self.branch_name, self.branch_bus_order, \
        self.gen, self.gen_indexs, self.gen_order, self.gen_name, self.gen_bus_order = self.get_data_struct()
        self.gen_info = None
        self.branch_y, self.y_matrix = None, None
        self.get_branch_y()
        self.get_Y_matrix()
        self.mva_base = 100

        self.describe()

    def describe(self):
        """汇报当前的dat长啥样"""
        pass

    def get_data_struct(self):
        """
        目标是建立起【不依赖文件结构的】电网对象的索引，包括：
        1. 全集: bus, branch
        2. 根据名称组织起来的卡片: bus_indexs, branch_indexs
        3. 根据出现次序给定的顺序：bus_order， branch_order
        4. 将所有bus，branch对象按照此顺序标上序号
        @note: 此后出现的数据结构里，只有order字段，没有name字段。
        @return:
        """
        bus_indexs, branch_indexs, bus_order, branch_order, gen_indexs, gen_order = {}, {}, {}, {}, {}, {}
        bus_name, branch_name, gen_name, gen_bus_order, branch_bus_order = [], [], [], [], []

        bus = [self.cards[f]
               for f in self.cdt
               if self.cards[f].type in DAT.bus_card_set]
        gen = []
        n_bus = 0
        n_gen = 0
        for i, b in enumerate(bus):
            if b.name in bus_indexs:
                bus_indexs[b.name] += [b]
            else:
                bus_indexs[b.name] = [b]
                bus_order[b.name] = n_bus
                bus_name += [b.name]
                n_bus += 1
            b.order = bus_order[b.name]
            if _bus_has_gen(b):
                g_name = b.name + ' '
                gen_indexs[g_name] = [b]  # 默认了每个节点只有一台发电机
                gen_order[g_name] = n_gen
                gen_name += [g_name]
                gen_bus_order += [bus_order[b.name]]
                n_gen += 1

        branch = [self.cards[f]
                  for f in self.cdt
                  if self.cards[f].type in DAT.line_card_set]
        n_branch = 0
        for i, l in enumerate(branch):
            if l.name in bus_indexs:
                branch_indexs[l.name] += [l]
            else:
                branch_indexs[l.name] = [l]
                branch_order[l.name] = n_branch
                branch_bus_order.append([bus_order[l.name1], bus_order[l.name2]])
                branch_name += [l.name]
                n_branch += 1
            l.order = branch_order[l.name]
            l.order1 = bus_order[l.name1]
            l.order2 = bus_order[l.name2]

        branch_bus_order = np.array(branch_bus_order)
        return bus, bus_indexs, bus_order, bus_name, \
               branch, branch_indexs, branch_order, branch_name, branch_bus_order, \
               gen, gen_indexs, gen_order, gen_name, gen_bus_order

    def __str__(self):
        """将结构化的card还原回str行"""
        return '\n'.join([x if type(x) is str else str(x) for x in self.cards]) + '\n'

    def read_pfo(self, dat_path):
        n_bus = len(self.bus_order)

        # 读pfo文件
        pfo_lines = open(dat_path[:-4] + '.pfo', 'rb').readlines()

        # 检查是否收敛：
        key = '不收敛'.encode('gbk')
        for l in pfo_lines:
            if key in l:
                return None

        # 只提取 bus_info_start_flag 那一段
        stl = [i for i, l in enumerate(pfo_lines) if '*  节点相关数据列表'.encode('gbk') in l]
        # 有些潮流结果有问题，但是又不想让程序中断
        if len(stl) == 0:
            return None
        stl = stl[0]
        edl = [i for i, l in enumerate(pfo_lines[stl:]) if '------'.encode('gbk') in l][1] + stl
        r = pfo_lines[stl:edl]
        sstl = [i for i, l in enumerate(r) if b'\r\n' == l][0] + 1
        r2 = r[sstl:]
        for i in range(len(r2)):
            s = r2[i]
            if s[45:46] == b' ':  # 'PF'
                s = s[:45] + b'0' + s[46:]
            if s[97:98] == b' ':  # 'TYPE'
                s = s[:97] + b'*' + s[98:]
            if s[102:103] == b' ':  # 'OWNER'  @todo 注意所有者代码是3个字节，因此这里应该用完全的三个字节进行比较。
                s = s[:102] + b'*' + s[103:]
            if s[108:110] == b'  ':  # 'ZONE'
                s = s[:108] + b'* ' + s[109:]
            r2[i] = s[:-2].decode('gbk', errors='ignore')
        df = pd.DataFrame(r2)[0].str.split(r'\s+|\s*/\s*|\n', expand=True).drop([0], axis=1)

        # 重命名、数据类型转换
        df.columns = [BUS_NAME_STR, BUS_BASE_STR, 'V',
                      GEN_P, GEN_Q, 'PF', LOAD_P, LOAD_Q,
                      'SHUNTQ1', 'SHUNTQ2', 'SHUNTQ3',
                      'TYPE', 'OWNER', 'ZONE', VOLTAGE, ANGLE]
        df[[BUS_NAME_STR, 'TYPE', 'OWNER', 'ZONE']] = df[[BUS_NAME_STR, 'TYPE', 'OWNER', 'ZONE']].astype('str',
                                                                                                         copy=False)
        df[[BUS_BASE_STR, 'V', GEN_P, GEN_Q, 'PF', LOAD_P, LOAD_Q, 'SHUNTQ1', 'SHUNTQ2', 'SHUNTQ3', VOLTAGE, ANGLE]] = \
            df[[BUS_BASE_STR, 'V', GEN_P, GEN_Q, 'PF', LOAD_P, LOAD_Q, 'SHUNTQ1', 'SHUNTQ2', 'SHUNTQ3', VOLTAGE,
                ANGLE]].astype('float', copy=False)
        df.loc[df['TYPE'] == '*', 'TYPE'] = ' '
        df.loc[df['OWNER'] == '*', 'OWNER'] = ' '
        df.loc[df['ZONE'] == '*', 'ZONE'] = ' '

        # 标幺值转换
        df[[GEN_P, GEN_Q, 'PF', LOAD_P, LOAD_Q, 'SHUNTQ1', 'SHUNTQ2', 'SHUNTQ3']].div(self.mva_base)
        df['SHUNTB'] = df['SHUNTQ1'].copy().add(df['SHUNTQ3']).div(df[VOLTAGE])
        df[BUS_NAME_STR] = list(map(BNameFormatter.get_name, df[BUS_NAME_STR], df[BUS_BASE_STR]))

        df.index = [self.bus_order[name] for name in df[BUS_NAME_STR]]

        # todo
        #  1. 并不确定如何处理shuntQ
        #  2. 并不确定如何处理恒功率负荷

        return df.sort_index()

    def get_branch_y(self):
        if self.branch_y is not None:
            return self.branch_y
        n_line = len(self.branch_order)
        y = np.zeros([n_line, 4], dtype=np.complex_)
        for l in self.branch:
            y[l.order, :] += l.get_y()
        self.branch_y = y
        return self.branch_y

    def get_Y_matrix(self):
        if self.y_matrix is not None:
            return self.y_matrix
        # 需要 bus_dict{name: index}, line_dict
        n_bus = len(self.bus_order)
        n_line = len(self.branch_order)

        y = np.zeros([n_bus, n_bus], dtype=np.complex_)
        by = self.get_branch_y()
        for l in self.branch:
            try:
                idx = l.order
                bus1 = l.order1
                bus2 = l.order2
                y[bus1, bus1] += by[idx, 0]
                y[bus1, bus2] += by[idx, 1]
                y[bus2, bus1] += by[idx, 2]
                y[bus2, bus2] += by[idx, 3]
            except ValueError:
                print(l.name, ' is ignored. Line: ', str(l))

        self.y_matrix = y

        coo_mask = [[i, i] for i in range(n_bus)]
        coo_mask.extend([[self.branch_indexs[l][0].order1, self.branch_indexs[l][0].order2]
                         for l in self.branch_name])
        coo_mask.extend([[self.branch_indexs[l][0].order2, self.branch_indexs[l][0].order1]
                         for l in self.branch_name])

        self.coo_mask = np.array(coo_mask).transpose()
        self.line_loc_in_coo = np.array([[n_bus + i, n_bus + n_line + i] for i in range(n_line)])

        return self.y_matrix

    @staticmethod
    def build_from_folder(folder_path):
        # todo 补充folder_path == 。dat的情况
        for f in os.listdir(folder_path):
            if '.dat' in f.lower():
                lines = open(folder_path + '/' + f, 'rb').readlines()
                return DAT.build_from_lines(lines)
        raise ValueError('cannot find .dat file in folder_path!')

    @staticmethod
    def build_from_lines(ll: List[str]):

        lines = []
        can = []
        cc1 = []
        cdt = []
        cc2 = []

        position = 1

        ll = [bpa_card_line(line) for line in ll]
        for i, l in enumerate(ll):
            if l == b'\n':
                lines.append('\n')
                continue

            c = _to_card(l, bpa_annotation.card_types)
            if c:
                lines.append(c)
                can.append(i)
                continue

            c = _to_card(l, bpa_dat_before_data.card_types)
            if c:
                lines.append(c)
                cc1.append(i)
                continue

            c = _to_card(l, bpa_dat_data.card_types)
            if c and position:
                lines.append(c)
                cdt.append(i)
                continue
            else:
                position = 0

            c = _to_card(l, bpa_dat_before_data.card_types)
            if c:
                lines.append(c)
                cc2.append(i)
                continue

            # raise ValueError('unknown bline!')
            print('unknown bline!: ', l)
            position = 1

        return DAT(lines, can, cc1, cdt, cc2)


def _bus_has_gen(bus: _OneNameUid):
    """
    由于dat文件没有明示哪个bus有GEN，于是只能自己判断是否有GEN
    @param bus:
    @return:
    """
    if (GEN_P in bus.field_index and bus.get_value(GEN_P)) \
            or (GEN_P_MAX in bus.field_index and bus.get_value(GEN_P_MAX)) \
            or (GEN_Q_MIN in bus.field_index and bus.get_value(GEN_Q_MIN)) \
            or (GEN_Q_MAX in bus.field_index and bus.get_value(GEN_Q_MAX)):
        return True
    return False


if __name__ == '__main__':
    # a = SwiG('G  bus36   100     5                 5           5            bus39   100')
    folder_path = r'D:\OneDrive\桌面\PsdEdit\a/0_100_0'
    # folder_path = r'E:\Data\transient_stability\300\dats\2_95_298_71_2_66'
    pfo_path = r'E:\Data\transient_stability\300\dats\2_95_298_71_2_66\2_95_298_71_2_66.pfo'
    pfo_path = r'D:\OneDrive\桌面\PsdEdit\a/0_100_0/0_100_0.pfo'
    # folder_path = r'D:\OneDrive\桌面\总调项目\20191112100152_save_1'
    dat = DAT.build_from_folder(folder_path)
    pfo = dat.read_pfo(pfo_path)
    y = dat.get_Y_matrix()

    # for f in os.listdir(folder_path):
    #     if '.dat' in f.lower():
    #         cards = open(folder_path + '/' + f, 'r', errors='ignore').readlines()
    #         break
    #
    # diff = [i for i, l in enumerate(b.cards) if str(l)[:-1].strip() != cards[i][:-1].strip()]  # \n位置不同，逗号后有空格
