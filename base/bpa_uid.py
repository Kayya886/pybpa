# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_uid
# Author:       MilkyDesk
# Date:         2021/7/6 11:37
# Description:
#   bpa中card类对象的唯一标识符
#   每个card类都需要继承
"""
import numpy as np

from base.bpa_base import _Card, _Field

# ------------------------- params --------------------------
BUS_NAME_STR = 'NAME'
BUS_BASE_STR = 'BASE'
BUS_NAME1_STR = 'NAME1'
BUS_BASE1_STR = 'BASE1'
BUS_NAME2_STR = 'NAME2'
BUS_BASE2_STR = 'BASE2'
GEN_ID = 'ID'
CKT_ID = 'CKT ID'  # 并联线路
ANGLE = 'ANGLE'  # 节点或发电机角度
VOLTAGE = 'VOLTAGE'  # 电压幅值
ORDER = 'ORDER'  # 特指dat中的bus_order, line_order
GEN_P_MAX = 'GEN_P_MAX'
GEN_P = 'GEN_P'
GEN_Q = 'GEN_Q'
GEN_Q_MAX = 'GEN_Q_MAX'
GEN_Q_MIN = 'GEN_Q_MIN'
LOAD_P = 'LOAD_P'
LOAD_Q = 'LOAD_Q'

# ------------------------- params --------------------------
class BNameFormatter:
    name_f = _Field(1, 8, '', 'A8')
    base_f = _Field(1, 4, '', 'F4.0')

    @staticmethod
    def get_name(name: str, base: float):
        return BNameFormatter.name_f.write(name) + BNameFormatter.base_f.write(base)

    @staticmethod
    def parse_name(name: str):
        assert len(name) == 14
        return BNameFormatter.name_f.read(name[:-4]), BNameFormatter.base_f.read(name[-4:])

class LNameFormatter:

    @staticmethod
    def get_name(name1: str, base1: float, name2: str, base2: float, ckt: str):
        return BNameFormatter.get_name(name1, base1) + BNameFormatter.get_name(name2, base2) + ckt

    @staticmethod
    def parse_name(name: str):
        s = bytes(name, encoding='gbk')
        assert len(s) == 29
        n1, b1 = BNameFormatter.parse_name(s[:14].decode('gbk', errors='ignore'))
        n2, b2 = BNameFormatter.parse_name(s[14:28].decode('gbk', errors='ignore'))
        ckt = s[28:]
        return n1, b1, n2, b2, ckt

    @staticmethod
    def parse_B(name: str, ckt=False):
        if ckt:
            return name[:14], name[14:28], name[28:]
        else:
            return name[:14], name[14:28]

class GNameFormatter:

    @staticmethod
    def get_name(name: str, base: float, id: str):
        return BNameFormatter.get_name(name, base) + id

    @staticmethod
    def parse_name(name: str):
        assert len(name) == 15
        n, b = BNameFormatter.parse_name(name[:-1])
        id = name[-1:]
        return n, b, id

    @staticmethod
    def parse_B(name, id=False):
        if id:
            return name[:-1], name[-1:]
        else:
            return name[:-1]


class _TypeNameUid(_Card):
    def __init__(self, bline):
        super(_TypeNameUid, self).__init__(bline)
        self.name = self.fields[0].default


class _OneNameUid(_Card):
    def __init__(self, bline):
        super(_OneNameUid, self).__init__(bline)
        self.order = None
        # try 是对当前各种父类fields编写不完整的补丁
        try:
            n, b = self.field_index[BUS_NAME_STR], self.field_index[BUS_BASE_STR]
            self.name = BNameFormatter.get_name(self.values[n], self.values[b])
        except:
            self.name = self.fields[0].default
            print('line cannot parse name: ', bline.decode('gbk', errors='ignore'))


class _TwoNameUid(_Card):
    def __init__(self, bline):
        super(_TwoNameUid, self).__init__(bline)
        self.order = None
        self.order1, self.order2 = None, None
        # try 是对当前各种父类fields编写不完整的补丁
        try:
            n1, b1 = self.field_index[BUS_NAME1_STR], self.field_index[BUS_BASE1_STR]
            n2, b2 = self.field_index[BUS_NAME2_STR], self.field_index[BUS_BASE2_STR]
            ckt = self.field_index[CKT_ID]
            self.name1 = BNameFormatter.get_name(self.values[n1], self.values[b1])
            self.name2 = BNameFormatter.get_name(self.values[n2], self.values[b2])
            self.name = LNameFormatter.get_name(self.values[n1], self.values[b1], self.values[n2], self.values[b2], self.values[ckt])
        except:
            print('line cannot parse name: ', bline.decode('gbk', errors='ignore'))
            raise ValueError('有人需要有两个名字，但是他没有名字。')


class _GenUid(_Card):
    def __init__(self, bline):
        super(_GenUid, self).__init__(bline)
        self.order = None
        self.order1 = None
        # try 是对当前各种父类fields编写不完整的补丁
        try:
            n, b, id = self.field_index[BUS_NAME_STR], self.field_index[BUS_BASE_STR], self.field_index[GEN_ID]
            self.name = GNameFormatter.get_name(self.values[n], self.values[b], self.values[id])
            self.name1 = GNameFormatter.parse_B(self.name)
        except:
            self.name = self.fields[0].default


class _LineCard(_TwoNameUid):
    def get_y(self):
        """
        返回自身对导纳矩阵的影响(L,E,T,TP)
        特别的卡（L+，LD，LM，R，RZ）需要自己实现，否则会报错
        """
        # 兼容不对称卡E
        if 'G' in self.field_index:
            y11 = self.get_value('G') + 1j * self.get_value('B')
        elif 'G1' in self.field_index:
            y11 = self.get_value('G1') + 1j * self.get_value('B1')
            y22 = self.get_value('G2') + 1j * self.get_value('B2')

        # 兼容变压器T,TP
        if 'TP1' in self.field_index:
            k = (self.get_value('TP2') * self.get_value(BUS_BASE1_STR)) / (self.get_value('TP1') * self.get_value(BUS_BASE2_STR)) + 0j
            y22 = 0
        elif 'PHASE SHIFT DEG' in self.field_index:
            k = np.exp(-1j * np.pi / 180 * self.get_value('PHASE SHIFT DEG'))
            y22 = 0
        else:
            k = 1 + 0j
            y22 = y11

        y = 1 / (self.get_value('R') + 1j * self.get_value('X'))
        y12 = y / k
        y21 = y / k.conjugate()

        return y11 + y, -y12, -y21, y22 + (y / (abs(k) ** 2))