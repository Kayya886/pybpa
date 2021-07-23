# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_swi_output
# Author:       MilkyDesk
# Date:         2021/6/28 16:27
# Description:
#   
"""
import sys
import inspect

from bpa_operations.bpa_base import _build, CARD_TYPE
from bpa_operations.bpa_uid import _TypeNameUid, _OneNameUid, _TwoNameUid, BUS_NAME_STR, BUS_BASE_STR, BUS_NAME1_STR, \
    BUS_NAME2_STR, BUS_BASE1_STR, BUS_BASE2_STR, _GenUid, CKT_ID, ANGLE, VOLTAGE, LOAD_P, LOAD_Q, GEN_P, GEN_Q


# ======================================================================================================================
#
#                                             输出控制以90卡作为开始
#
# ======================================================================================================================


class Swi90(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', '90']])


class SwiMH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'MH'],
                                        [4, 5, 'NUMPLT', 'I2', 4],
                                        [58, 60, 'NSKP', 'I3'],
                                        [77, 77, 'IFPLOT', 'I1'],
                                        [78, 78, 'INOPERSIST', 'I1']])


class SwiBH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'BH'],
                                        [4, 4, 'TS', 'I1'],
                                        [5, 5, 'NDOT', 'I1', 4],
                                        [9, 9, 'CLASSIFICATION1', 'I1'],
                                        [13, 18, 'MAX1', 'F6.0'],
                                        [20, 25, 'MIN1', 'F6.0'],
                                        [27, 27, 'CLASSIFICATION2', 'I1'],
                                        [31, 36, 'MAX2', 'F6.0'],
                                        [38, 43, 'MIN2', 'F6.0'],
                                        [45, 45, 'CLASSIFICATION3', 'I1'],
                                        [49, 54, 'MAX3', 'F6.0'],
                                        [56, 61, 'MIN3', 'F6.0'],
                                        [63, 63, 'CLASSIFICATION4', 'I1'],
                                        [67, 72, 'MAX4', 'F6.0'],
                                        [74, 79, 'MIN4', 'F6.0']])


class SwiB(_OneNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'B'],
                                        [4, 11, BUS_NAME_STR, 'A8'],
                                        [12, 15, BUS_BASE_STR, 'F4.0', None, '单位：kV'],
                                        [18, 18, VOLTAGE, 'I1'],
                                        [19, 19, ANGLE, 'I1'],
                                        [21, 21, 'FREQUENCY', 'I1'],
                                        [24, 24, LOAD_P, 'I1'],
                                        [27, 27, LOAD_Q, 'I1'],
                                        [32, 34, 'GRP', 'I3'],
                                        [38, 38, 'A VOLTAGE', 'I1'],
                                        [39, 39, 'A ANGLE', 'I1'],
                                        [40, 40, 'B VOLTAGE', 'I1'],
                                        [41, 41, 'B ANGLE', 'I1'],
                                        [42, 42, 'C VOLTAGE', 'I1'],
                                        [43, 43, 'C ANGLE', 'I1'],
                                        [44, 44, 'NEG VOLTAGE', 'I1'],
                                        [45, 45, 'NEG VOL ANGLE', 'I1'],
                                        [46, 46, 'ZERO VOLTAGE', 'I1'],
                                        [47, 47, 'ZERO VOL ANGLE', 'I1']])


class SwiGH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'GH'],
                                        [5, 5, 'NDOT', 'I1', 4],
                                        [7, 14, 'REFBUS', 'A8'],
                                        [15, 18, 'BASE', 'F4.0'],
                                        [20, 20, 'REFERENCE ID', 'A1'],
                                        [22, 23, 'CLASSIFICATION1', 'I2'],
                                        [27, 32, 'MAX1', 'F6.0'],
                                        [34, 39, 'MIN1', 'F6.0'],
                                        [41, 42, 'CLASSIFICATION2', 'I2'],
                                        [46, 51, 'MAX2', 'F6.0'],
                                        [53, 58, 'MIN2', 'F6.0'],
                                        [60, 61, 'CLASSIFICATION3', 'I2'],
                                        [65, 70, 'MAX3', 'F6.0'],
                                        [72, 77, 'MIN3', 'F6.0']])


class SwiGHC(_TypeNameUid):
    fields, line, field_index = _build([[1, 3, CARD_TYPE, 'A3', 'GHC'],
                                        [6, 7, 'CLASSIFICATION1', 'I2'],
                                        [11, 16, 'MAX1', 'F6.0'],
                                        [18, 23, 'MIN1', 'F6.0'],
                                        [25, 26, 'CLASSIFICATION2', 'I2'],
                                        [30, 35, 'MAX2', 'F6.0'],
                                        [37, 42, 'MIN2', 'F6.0'],
                                        [44, 45, 'CLASSIFICATION3', 'I2'],
                                        [49, 54, 'MAX3', 'F6.0'],
                                        [56, 61, 'MIN3', 'F6.0'],
                                        [63, 64, 'CLASSIFICATION4', 'I2'],
                                        [68, 73, 'MAX4', 'F6.0'],
                                        [75, 80, 'MIN4', 'F6.0']])


class SwiG(_GenUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'G'],
                                        [4, 11, BUS_NAME_STR, 'A8'],
                                        [12, 15, BUS_BASE_STR, 'F4.0', None, '单位：?'],
                                        [17, 17, 'ID', 'A1'],
                                        [20, 20, ANGLE, 'I1'],
                                        [23, 23, 'VELOCITY DEVIATION', 'I1'],
                                        [26, 26, 'FIELD VOLTS', 'I1', None, '励磁电压'],
                                        [29, 29, 'FLUX LINKAGE EQP', 'I1'],
                                        [32, 32, 'MAIN FIELD SET', 'I1'],
                                        [35, 35, 'MACH POWER', 'I1'],
                                        [38, 38, GEN_P, 'I1'],  # todo 发电机电磁功率不等于发电机有功..下面GEN_Q同理
                                        [41, 41, 'EXCITER SAT', 'I1'],
                                        [44, 44, 'REGULATOR OUTPUT', 'I1'],
                                        [47, 47, 'ACCELERATING POWER', 'I1'],
                                        [50, 50, GEN_Q, 'I1'],
                                        [53, 53, 'EXCITER SUP.SIG', 'I1'],
                                        [56, 56, 'DAMPING TORQUE', 'I1'],
                                        [59, 59, 'FIELD CURRENT', 'I1'],
                                        [63, 70, 'GNAME', 'A8'],
                                        [71, 74, 'GBSE', 'F4.0'],
                                        [75, 75, 'REFERENCE ID', 'I1'],
                                        [76, 78, 'GRP', 'I3'],
                                        [80, 80, 'IS SWING', 'I1', None, '存疑，说明书和卡对不上， P409']])


class SwiGP(_GenUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'G+'],
                                        [4, 11, BUS_NAME_STR, 'A8'],
                                        [12, 15, BUS_BASE_STR, 'F4.0', None, '单位：?'],
                                        [17, 17, 'ID', 'A1'],
                                        [19, 19, 'VOL', 'I1'],
                                        [21, 21, 'CURRENT', 'I1'],
                                        [23, 23, 'EQ', 'I1'],
                                        [25, 25, 'EDP', 'I1'],
                                        [27, 27, 'EQP', 'I1'],
                                        [29, 29, 'EDPP', 'I1'],
                                        [31, 31, 'EQPP', 'I1'],
                                        [33, 33, 'SAT D', 'I1'],
                                        [35, 35, 'SAT Q', 'I1'],
                                        [37, 37, 'XD MOD', 'I1'],
                                        [39, 39, 'XQ MOD', 'I1'],
                                        [41, 41, 'FD', 'I1'],
                                        [43, 43, 'FQ', 'I1'],
                                        [45, 45, 'R', 'I1'],
                                        [47, 47, 'X', 'I1'],
                                        [50, 50, 'GENERATOR MVR', 'I1'],
                                        [53, 53, 'EXCITER SUP.SIG', 'I1'],
                                        [56, 56, 'DAMPING TORQUE', 'I1'],
                                        [59, 59, 'FIELD CURRENT', 'I1'],
                                        [63, 70, 'GNAME', 'A8'],
                                        [71, 74, 'GBSE', 'F4.0'],
                                        [75, 75, 'REFERENCE ID', 'I1'],
                                        [76, 78, 'GRP', 'I3'],
                                        [80, 80, 'IS SWING', 'I1', None, '存疑，说明书和卡对不上， P409']])


class SwiLH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'LH'],
                                        [5, 5, 'CLASSIFICATION1', 'I1'],
                                        [9, 14, 'MAX1', 'F6.0'],
                                        [16, 21, 'MIN1', 'F6.0'],
                                        [23, 23, 'CLASSIFICATION2', 'I1'],
                                        [27, 32, 'MAX2', 'F6.0'],
                                        [34, 39, 'MIN2', 'F6.0'],
                                        [41, 41, 'CLASSIFICATION3', 'I1'],
                                        [45, 50, 'MAX3', 'F6.0'],
                                        [52, 57, 'MIN3', 'F6.0'],
                                        [59, 61, 'NSYM', 'I3', 4],
                                        [63, 65, 'NTIM', 'I3', 4],
                                        [67, 70, 'TD', 'F4.3', 0.012],
                                        [72, 75, 'T1', 'F4.3', 0.053],
                                        [77, 80, 'T2', 'F4.3', 0.053]])


class SwiL(_TwoNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'L'],
                                        [4, 11, BUS_NAME1_STR, 'A8'],
                                        [12, 15, BUS_BASE1_STR, 'F4.0'],
                                        [17, 24, BUS_NAME2_STR, 'A8'],
                                        [25, 28, BUS_BASE2_STR, 'F4.0'],
                                        [30, 30, CKT_ID, 'A1'],
                                        [35, 35, 'LINE FLOW MW', 'I1'],
                                        [38, 38, 'LINE FLOW MVA', 'I1'],
                                        [41, 41, 'APPARENT IMPEDANCE', 'I1'],
                                        [44, 44, 'IMPEDANCE ANGLE', 'I1'],
                                        [72, 74, 'GRP', 'I3'],
                                        [80, 80, '???', 'I1', 0, '非零，输出线路功率曲线的衰减系数、振荡频率和阻尼比']])


class SwiLC(_TwoNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'LC'],
                                        [4, 11, BUS_NAME1_STR, 'A8'],
                                        [12, 15, BUS_BASE1_STR, 'F4.0'],
                                        [17, 24, BUS_NAME2_STR, 'A8'],
                                        [25, 28, BUS_BASE2_STR, 'F4.0'],
                                        [30, 30, CKT_ID, 'A1'],
                                        [33, 33, 'DEL ANGLE', 'I1'],
                                        [35, 35, 'LINE FLOW AMPS', 'I1'],
                                        [38, 38, 'Z-Z', 'I1'],
                                        [41, 41, 'R-R', 'I1'],
                                        [49, 49, 'A', 'I1'],
                                        [56, 56, 'B', 'I1'],
                                        [63, 63, 'C', 'I1'],
                                        [70, 70, 'N', 'I1'],
                                        [71, 71, 'Z', 'I1'],
                                        [72, 74, 'GRP', 'I3']])


class SwiDH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'DH']])


class SwiD(_TwoNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'D']])


class SwiRH(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'RH']])


class SwiR(_TwoNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'R']])


class SwiPY(_TwoNameUid):
    """实际上不是ILineCard范围"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'PY']])


class SwiOBV(_TypeNameUid):
    fields, line, field_index = _build([[1, 3, CARD_TYPE, 'A3', 'OBV']])


class SwiOGM(_TypeNameUid):
    fields, line, field_index = _build([[1, 3, CARD_TYPE, 'A3', 'OGM']])


class SwiOLT(_TypeNameUid):
    fields, line, field_index = _build([[1, 3, CARD_TYPE, 'A3', 'OLT']])


class Swi99(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', '99']])

# ======================================================================================================================
#
#                                               输出控制部分以99卡作为结束
#
# ======================================================================================================================

# ======================================================================================================================

card_types = {cls[1].fields[cls[1].field_index[CARD_TYPE]].default: cls[1]
              for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
              if '_' not in cls[0]}

# ======================================================================================================================
