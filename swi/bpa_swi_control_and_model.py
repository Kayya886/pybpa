# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_swi_control_and_model
# Author:       MilkyDesk
# Date:         2021/6/28 16:24
# Description:
#   
"""

import sys
import inspect

from bpa_operations.bpa_base import _build, CARD_TYPE
from bpa_operations.bpa_uid import _TypeNameUid, _OneNameUid, _TwoNameUid, BUS_NAME_STR, BUS_BASE_STR, BUS_NAME1_STR, \
    BUS_NAME2_STR, BUS_BASE1_STR, BUS_BASE2_STR, _GenUid


# ======================================================================================================================
#
#                                               PARA_FILE / CASE卡，第一个出现
#
# ======================================================================================================================


class SwiPARAFILE(_TypeNameUid):
    fields, line, field_index = _build([[1, 9, CARD_TYPE, 'A9', 'PARA_FILE'],
                                        [11, 80, '稳定参数文件名称', 'A70']])


class SwiCASE(_TypeNameUid):
    fields, line, field_index = _build([[1, 4, CARD_TYPE, 'A4', 'CASE'],
                                        [6, 15, 'PFCASE', 'A10'],
                                        [17, 17, 'INOPERSIST', 'I1'],
                                        [24, 24, 'IDERIV_CHECK', 'I1'],
                                        [30, 30, 'IWG_NO_LIM', 'I1'],
                                        [45, 49, 'X2FAC', 'F5.5', 0.65],
                                        [50, 54, 'XFACT', 'F5.5'],
                                        [55, 59, 'TDODPS', 'F5.5', 0.03],
                                        [60, 64, 'TQODPS', 'F5.5', 0.05],
                                        [65, 69, 'TDODPH', 'F5.5', 0.04],
                                        [70, 74, 'TQODPH', 'F5.5', 0.3],
                                        [75, 80, 'CFACL2', 'F6.5', 0.36, '负序负荷导纳标么值Y2*＝0.19+jCFACL2']])


class SwiF0(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'F0'],
                                        [5, 5, 'IG', 'I1'],
                                        [8, 8, 'IA', 'I1'],
                                        [10, 17, 'GEN1', 'A8'],
                                        [18, 21, 'BASE1', 'F4.0', None, 'kV'],
                                        [22, 22, 'ID', 'A1'],
                                        [24, 31, 'GEN2', 'A8'],
                                        [32, 35, 'BASE2', 'F4.0', None, 'kV'],
                                        [36, 36, 'ID', 'A1'],
                                        [38, 42, 'AMAX', 'F5.0', 200],
                                        [43, 47, 'AMIN', 'F5.0', -200],
                                        [50, 50, 'IV', 'I1', 0],
                                        [52, 59, 'VOL BUS', 'A8'],
                                        [60, 63, 'VOL BASE', 'F4.0'],
                                        [68, 75, 'FRQ BUS', 'A8'],
                                        [76, 79, 'FRQ BASE', 'F4.0']])


class SwiF1(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'F1'],
                                        [5, 9, 'TBUSF', 'F5.4', 0],
                                        [20, 22, 'DMPALL', 'F3.2', 0],
                                        [26, 26, 'IAMRTS', 'I1'],
                                        [28, 31, 'VCHGL', 'F4.3', 0.5, 'P.U.'],
                                        [33, 33, 'SP TO SG', 'I1'],
                                        [37, 41, 'GSIQU', 'F5.5'],
                                        [51, 51, 'VRLIM', 'I1']])


class SwiLS(_TwoNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'LS'],
                                        [3, 80, '待补充', 'A78']])


class SwiFLT(_TwoNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 3, CARD_TYPE, 'A3', 'FLT'],
                                        [4, 80, '待补充', 'A77']])


class SwiMC(_TypeNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'MC'],
                                        [3, 80, '待补充', 'A78']])


class SwiMF(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'MF'],
                                        [3, 80, '待补充', 'A78']])


class SwiMG(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'MG'],
                                        [3, 80, '待补充', 'A78']])


class SwiG(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'G'],
                                        [2, 80, '待补充', 'A79']])


class SwiM(_GenUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'M'],
                                        [2, 80, '待补充', 'A79']])


class SwiL(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'L'],
                                        [2, 80, '待补充', 'A79']])


class SwiLN(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'LN'],
                                        [3, 80, '待补充', 'A78']])

class SwiX(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'X'],
                                        [2, 80, '待补充', 'A79']])


class SwiD(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'D'],
                                        [2, 80, '待补充', 'A79']])


class SwiE(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'E'],
                                        [2, 80, '待补充', 'A79']])


class SwiF(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'T'],
                                        [2, 80, '待补充', 'A79']])


class SwiT(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'F'],
                                        [2, 80, '待补充', 'A79']])


class SwiO(_OneNameUid):
    """NOTE, OEN,OEX, O*"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'O'],
                                        [2, 80, '待补充', 'A79']])


class SwiS(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'S'],
                                        [2, 80, '待补充', 'A79']])


class SwiI(_OneNameUid):
    """NOTE"""
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', 'I'],
                                        [2, 80, '待补充', 'A79']])


# class SwiGG(_Card):
#     """NOTE: G名称和输出卡有很多重复，因此需要重新定义。"""
#     fields, bline, field_index = _build([[1, 2, 'GG', 'A2', 'GG'],
#                                         [3, 80, '待补充', 'A78']])
#
# class SwiLS(_Card):
#     """NOTE"""
#     fields, bline, field_index = _build([[1, 2, 'LS', 'A2', 'LS'],
#                                         [3, 80, '待补充', 'A78']])


class SwiFF(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'FF'],
                                        [5, 7, 'T', 'F3.0'],
                                        [9, 11, 'DT', 'F3.0', 1],
                                        [13, 17, 'ENDT', 'F5.0', 300],
                                        [19, 21, 'DTC', 'F3.1'],
                                        [23, 25, 'ISTP', 'I3'],
                                        [27, 31, 'TOLI', 'F5.5', 0.01, 'P.U.'],
                                        [33, 35, 'ILIM', 'I3', 25],
                                        [37, 40, 'DELANG', 'F4.4', 0.1, 'DEGREE'],
                                        [42, 43, 'DTDV', 'I2', 8],
                                        [45, 47, 'DMPMLT', 'F3.3', 1],
                                        [55, 56, 'FRQBSE', 'F2.0', 50, 'Hz'],
                                        [58, 58, 'LOVTEX', 'I1', 0, '为1时表示无低电压检查（低于0.4pu），否则进行检查'],
                                        [60, 60, 'IMBLOK', 'I1', 1, '马达滑差达到1时的处理方法'],
                                        [64, 64, 'MFDEP', 'I1'],
                                        [65, 65, 'IGSLIM', 'I1', 0],
                                        [66, 66, 'LSOLQIT', 'I1'],
                                        [68, 68, 'NOANGLIM', 'I1'],
                                        [70, 70, 'INFBUS', 'I1'],
                                        [71, 71, 'NOPP', 'I1'],
                                        [72, 72, 'NODQ', 'I1'],
                                        [73, 73, 'NOSAT', 'I1'],
                                        [74, 74, 'NOGV', 'I1'],
                                        [75, 75, 'IEQPC', 'I1'],
                                        [76, 76, 'NOEX', 'I1'],
                                        [77, 77, 'MF TO MG', 'I1'],
                                        [78, 78, 'NOSC', 'I1'],
                                        [79, 79, 'MG TO MG', 'I1'],
                                        [80, 80, 'NOLOAD', 'I1']])


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
