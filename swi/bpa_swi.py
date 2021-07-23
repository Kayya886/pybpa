"""
# Name:         transient_stability: bpa_swi
# Author:       MilkyDesk
# Date:         2021/6/25 11:19
# Description:
#   定义swi类
"""

# ------------------------- fields --------------------------
import os

from typing import List

import pandas as pd

from bpa_operations.bpa_uid import _OneNameUid, _TwoNameUid, BUS_BASE_STR, BUS_NAME_STR, ANGLE, VOLTAGE
from bpa_operations.bpa_base import _to_card, _build, CARD_TYPE, _Field
from bpa_operations.bpa_str import bpa_card_line
from bpa_operations.dat.bpa_dat import DAT
from bpa_operations.swi import bpa_swi_control_and_model as CM, bpa_swi_output as O
from bpa_operations import bpa_annotation as A
from bpa_operations.bpa_uid import *


# ------------------------- fields --------------------------

class SWI:
    """
    定位：
    提供Swi_Card的管理工具
    行为：
    1. 初始化: 读取文件
    2. __str__: 反向文本化功能。应该与_Card的组织类似。
    """
    annotation_card = {'.', ' ', 'C'}
    control_model_card = {'PARA_FILE', 'CASE', 'F0', 'F1', 'FF', 'LS', 'FLT', 'M', 'E', 'F', 'O', 'S', 'I', 'LN', 'G'}
    output_card = {'90', 'B', 'BH', 'GH', 'GHC', 'G', 'GP', 'LH', 'L', 'LC', 'DH', 'D', 'RH', 'R', 'PY', 'OBV', 'OGM',
                   'OLT', '99'}

    control_format = 'I1'
    """swi文件输出卡的默认控制格式"""

    base_reader = _Field(1, 5, BUS_BASE_STR, 'F5.0', 0).read
    base_writer = _Field(1, 4, BUS_BASE_STR, 'F4.0', 0).write


    def __init__(self, *p):
        self.lines, self.can, self.ccm, self.cop, self.cf0, self.cf1, self.cff, self.cpara_file, self.c90, self.c99 = p
        self.swx_struct = self.get_output_struct()
        self.dat = None
        self.start_step = 19
        self.end_step = 40
        self.sim_step_mask = None
        self.time_index = [10.0, 15.0]  # lsd文件中的信息
        self.fault_step, self.clear_step = 10.0, 15.0

    def get_output_struct(self):
        """
        目标是得到输出文件的第几段，输出对象和输出字段
        似乎GH，BH卡的NDOT控制数据精度字段，对于读取程序来说并不会影响到？
        @return:
        """
        swx_struct = {'B': {}, 'L': {}, 'G': {}, 'D': {}, 'R': {}}
        for c in self.cop:
            card = self.lines[c]
            if 'H' in card.type or '9' in card.type:
                continue
            temp = [f.name
                    for f, v in zip(card.fields, card.values)
                    if f.format == SWI.control_format and v and v > 0]  # bpa_online竟然是1，bpa说明书是3
            if temp:
                if card.type in swx_struct:  # 此时是L卡
                    if card.name in swx_struct[card.type]:  # 如果LC卡信息已经在里面
                        swx_struct[card.type][card.name] = temp + swx_struct[card.type][card.name]
                    else:  # 如果LC卡信息还没在里面
                        swx_struct[card.type][card.name] = temp
                elif card.type == 'LC':  # 如果是LC卡
                    if card.name in swx_struct['L']:  # 如果L卡信息已经在里面
                        swx_struct['L'][card.name] = swx_struct['L'][card.name] + temp
                    else:  # 如果L卡信息还没在里面
                        swx_struct['L'][card.name] = temp
        return swx_struct

    def __str__(self):
        """将结构化的card还原回str行"""
        return '\n'.join([x if type(x) is str else str(x) for x in self.lines]) + '\n'

    def read_swx(self, swx_file, dat: DAT = None):
        # if not self.swx_struct:
        #     self.swx_struct = self.get_output_struct()
        if not self.dat:
            self.dat = dat
            if dat is None:
                raise ValueError('没有关联的Dat对象！')

        swx_lines = open(swx_file, 'r').readlines()
        swx_lines = [l[:-1] for l in swx_lines if l[0] == ' ']
        start_index = [i for i, l in enumerate(swx_lines) if '输出数据列表' in l]
        start_index.append(len(swx_lines))
        blocks = [swx_lines[i:j] for i, j in zip(start_index[:-1], start_index[1:])]

        # assert len(self.swx_struct) == len(blocks)
        swx, pf = {k: {} for k in self.swx_struct.keys()}, {k: {} for k in self.swx_struct.keys()}
        for b in blocks:
            t, name, idx, data1, data2 = self._parse_block(b)
            swx[t].update({idx: data1})
            pf[t].update({idx: data2})

        # 打标签，将字典转换为断面
        label = self.get_label(swx)
        pfs = self.swx2pf_section(dat, pf)

        # 截断和重新组织swx
        for k, v in swx.items():
            if v:
                for kk, vv in v.items():
                    v[kk] = vv.iloc[self.start_step: self.end_step, :]

        return swx, pfs, label

    def get_label(self, swx):
        # 功角稳定标签。默认所有发电机都有，毕竟发电机不完整的话，不叫功角稳定
        g = swx['G']
        max_delta_gen_angle, max_delta_sys_angle = None, None
        if g:
            temp_g = pd.concat([gg[ANGLE] for gg in g.values()], axis=1, ignore_index=True)
            temp_g.columns = g.keys()

            max_delta_gen_angle = temp_g.abs().max(axis=0)

            min_g = temp_g.min(axis=1)
            max_g = temp_g.max(axis=1)
            max_delta_sys_angle = max_g.sub(min_g, axis=0).abs().max(axis=0)

        # 电压稳定标签
        b = swx['B']
        min_bus_voltage, max_bus_voltage = None, None
        if b:
            temp_b = pd.concat([bb[VOLTAGE] for bb in b.values()], axis=1, ignore_index=True)
            temp_b.columns = b.keys()
            min_bus_voltage = temp_b.min(axis=0)
            max_bus_voltage = temp_b.max(axis=0)

        return {'G': [max_delta_gen_angle, max_delta_sys_angle],
                'B': [min_bus_voltage, max_bus_voltage]}

    def swx2pf_section(self, dat: DAT, swx_struct):
        """
        最终妥协成跟pfo一样的格式。
        @param dat:
        @param swx_struct:
        @return:
        """
        # for key, s in swx_struct.items():
        #     if s:
        #         # s里的假设所有df都一个样，随便取一个获取形状和index
        #         temp_df = list(s.values())[0]
        #         time_index = temp_df.index
        #         break
        pfs = {t: {} for t in self.time_index}

        for t in self.time_index:
            df_b = pd.DataFrame([swx_struct['B'][key].loc[t]
                               for key in swx_struct['B'].keys()], index=swx_struct['B'].keys())
            df_g = pd.DataFrame([swx_struct['G'][key].loc[t]
                                 for key in swx_struct['G'].keys()], index=swx_struct['G'].keys())
            df_g.index = [dat.gen_bus_order[i] for i in df_g.index]
            if ANGLE in df_g.columns:
                df_g.rename(columns={ANGLE: ANGLE + '_g'})
            df = pd.merge(df_b, df_g, how='outer', left_index=True, right_index=True)
            df.fillna(0, inplace=True)
            df.sort_index(inplace=True)
            pfs[t] = df

        return pfs

    def _parse_block(self, block):
        """

        @param block:
        @return: data1保留0-，data2为0+
        """
        info_line = block[0][(block[0].index(' * ') + 3):].split('\"')
        info_type = info_line[0]  # 发电机、母线、线路
        data = pd.DataFrame(block[4:])[0].str.split('\t', expand=True)

        # 解析元信息
        if '发电机' == info_type:
            t = 'G'
            name, idx = self._parse_gen(info_line[1])
        elif '节点' == info_type:
            t = 'B'
            name, idx = self._parse_bus(info_line[1])
        elif '线路' == info_type:
            t = 'L'
            name, idx = self._parse_2bus(info_line[1])
        elif '串联补偿' == info_type:
            t = 'R'
            name, idx = self._parse_2bus(info_line[1])
        elif '直流' == info_type:
            t = 'D'
            name, idx = self._parse_2bus(info_line[1])
        else:
            raise ValueError('info_type not in  {发电机，节点，线路，串联补偿，直流} ！')

        data.columns = ['STEP'] + self.swx_struct[t][name]
        data = data.astype('float')
        # 功率处理为标幺值
        for col in data.columns:
            if col in [GEN_Q, GEN_P, LOAD_Q, LOAD_P]:  # 暂时的，还有问题。todo
                data[col] = data[col].div(self.dat.mva_base)

        # 数据预处理,分割
        if self.sim_step_mask is None:
            self.sim_step_mask = (data['STEP'] - data['STEP'].shift(1)).astype('bool')

        # 需要处理发电机角度
        if t == 'G' and 'ANGLE' in data.columns:
            data['DANGLE'] = (data['ANGLE'] - data['ANGLE'].shift(1))
            data['DANGLE'][0] = 0
            assert len(data['DANGLE'][data['DANGLE'] == 180]) == 0, '出现时间步加速度恰好为180，检查是否有加速度超过180的！'
            data['DANGLE'][data['DANGLE'] > 180] = data['DANGLE'][data['DANGLE'] > 180] - 360
            data['DANGLE'][data['DANGLE'] <-180] = data['DANGLE'][data['DANGLE'] <-180] + 360
            data['ANGLE'] = data['DANGLE'].cumsum(axis=0) + data['ANGLE'][0]



        # 分割出t0+-的时刻
        data, data2 = data[self.sim_step_mask].copy(), data[~self.sim_step_mask].copy()
        data.set_index('STEP', inplace=True)
        data2.set_index('STEP', inplace=True)
        # self.fault_step, self.clear_step = data2.index[1], data2.index[2]

        return t, name, idx, data, data2.loc[1:]  # 舍去0时刻

    def get_sim_step(self):
        return self.cff.get_value('DT')

    def get_sim_len(self):
        return self.cff.get_value('ENDT')

    @staticmethod
    def build_from_folder(folder_path):
        for f in os.listdir(folder_path):
            if '.swi' in f.lower():
                lines = open(folder_path + '/' + f, 'rb').readlines()
                return SWI.build_from_lines(lines)
        raise ValueError('cannot find .swi file in folder_path!')

    @staticmethod
    def build_from_lines(ll: List[str]):
        ll = [bpa_card_line(line) for line in ll]

        lines = []
        can = []
        ccm = []
        cop = []

        position = 0
        # 唯一的卡
        cf0, cf1, cff, cpara_file, c90, c99 = [], [], [], [], [], []
        cm_only1 = {'F0': cf0, 'F1': cf1, 'FF': cff, 'PARA_FILE': cpara_file}
        op_only1 = {'90': c90, '99': c99}

        for i, l in enumerate(ll):
            if l == b'\n':
                lines.append('\n')
                continue

            c = _to_card(l, A.card_types)
            if c:
                lines.append(c)
                can.append(i)
                continue

            if l[:2] == b'90':
                position = 2

            if position < 2:
                c = _to_card(l, CM.card_types)
                if c:
                    position = 1
                    lines.append(c)
                    ccm.append(i)

                    if c.type in cm_only1:
                        cm_only1[c.type].append(i)

                    continue

            else:
                c = _to_card(l, O.card_types)
                if c:
                    position = 2
                    lines.append(c)
                    cop.append(i)

                    if c.type in op_only1:
                        op_only1[c.type].append(i)
                    continue

            print(0)
            raise ValueError('unknown bline!')

        for c in [cf0, cf1, cff, cpara_file, c90, c99]:
            if len(c) > 1:
                raise ValueError('has more than 1 card!')

        return SWI(lines, can, ccm, cop, cf0, cf1, cff, cpara_file, c90, c99)

    def _parse_bus(self, line):
        """不用uid的原因是，这里的base是F5.0"""
        bline = bytes(line, encoding='gbk') if type(line) is not bytes else line
        assert len(bline) == 14
        name = bline[:8].decode('gbk', errors='ignore') + SWI.base_writer(SWI.base_reader(bline[9:].decode('gbk', errors='ignore')))
        return name, self.dat.bus_order[name]

    def _parse_gen(self, line):
        """不用uid的原因是，这里的base是F5.0"""
        bline = bytes(line, encoding='gbk')
        assert len(bline) == 15
        name = bline[:8].decode('gbk', errors='ignore')\
               + SWI.base_writer(SWI.base_reader(bline[9:-1].decode('gbk', errors='ignore')))\
               + bline[-1:].decode('gbk', errors='ignore')
        return name, self.dat.gen_order[name]

    def _parse_2bus(self, line):
        """不用uid的原因是，这里的base是F5.0"""
        bline = bytes(line, encoding='gbk')
        assert len(bline) == 30
        name = self._parse_bus(bline[:14])[0] + self._parse_bus(bline[15:-1])[0] + line[-1]
        return name, self.dat.branch_order[name]


if __name__ == '__main__':
    # a = SwiG('G  bus36   100     5                 5           5            bus39   100')
    folder_path = r'E:\Data\transient_stability\300\bpa'
    # folder_path = r'D:\OneDrive\桌面\总调项目\20191112100152_save_1'
    op = '/1_100_113_0'
    dat = DAT.build_from_folder(folder_path + op)
    swi = SWI.build_from_folder(folder_path)
    # p = [i for i in os.listdir(folder_path + op) if '.swx' in i.lower()]
    # for f in p:
    #     print(f)
    #     swx, pfs, label  = swi.read_swx(folder_path + op + '/' + f, dat)
    swx, pfs, label  = swi.read_swx(r'E:/Data/transient_stability/300/bpa/0_105_0/0(115-69).SWX', dat)
    #         break
    # swx, pfs, label  = swi.read_swx(r'D:\OneDrive\桌面\PsdEdit\a\0_105_0\0(6-266).SWX', dat)
    # pfo_path = r'D:\OneDrive\桌面\PsdEdit\a\0_105_0\0_105_0.pfo'
    # pfo = dat.read_pfo(pfo_path)
