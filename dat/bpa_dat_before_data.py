# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_dat_before_data
# Author:       MilkyDesk
# Date:         2021/7/1 16:24
# Description:
#   
"""

import inspect
import sys

from base.bpa_base import _build, _Field, CARD_TYPE
from base.bpa_uid import _TypeNameUid, _OneNameUid, _TwoNameUid, BUS_NAME_STR, BUS_BASE_STR, BUS_NAME1_STR, BUS_NAME2_STR, BUS_BASE1_STR, BUS_BASE2_STR


class _DatControl(_TypeNameUid):
    NAME = 0

    def __init__(self, bline: str):
        super(_DatControl, self).__init__(bline)
        # todo 解析方式有风险 什么风险？
        self.l = bline.decode('gbk', errors='ignore').strip()  # 没有用到，仅作为debug
        l = [s.strip() for s in self.l[1:-1].split(',')]
        self.fields = [s.split('=')[0] for s in l]
        self.values = [s.split('=')[1] if '=' in s else '' for s in l]

    def __str__(self):
        return self.st + ', '.join([(self.fields[i] + ('=' + str(self.values[i]) if self.values[i] else ''))
                                    for i in range(len(self.fields))]) + self.ed


class DatControl1(_DatControl):
    st, ed = '(', ')'
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', st],
                                        [2, 80, '待定', 'A79']])


class DatControl2(_DatControl):
    st, ed = '/', '\\'
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', st],
                                        [2, 80, '待定', 'A79']])


class DatControl3(_DatControl):
    st, ed = '>', '<'
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', st],
                                        [2, 80, '待定', 'A79']])


class DatP(_TypeNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'P'],
                                        [2, 80, '待定', 'A79']])


class DatZ(_TypeNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'Z'],
                                        [2, 80, '待定', 'A79']])


class DatP(_TypeNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'P'],
                                        [2, 80, '待定', 'A79']])


class DatDZ(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'DZ'],
                                        [3, 80, '待定', 'A78']])


# ======================================================================================================================
#
#                                             第一部分以FF卡作为结束
#
# ======================================================================================================================

# ======================================================================================================================

card_types = {cls[1].fields[cls[1].field_index[CARD_TYPE]].default: cls[1]
              for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
              if '_' not in cls[0]}

# ======================================================================================================================