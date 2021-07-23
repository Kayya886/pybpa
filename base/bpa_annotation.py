# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_swi_annotation
# Author:       MilkyDesk
# Date:         2021/6/28 16:28
# Description:
#   
"""
import sys
import inspect

from base.bpa_base import _build, CARD_TYPE
from base.bpa_uid import _TypeNameUid

# ======================================================================================================================
#
#                                                   注释卡，可出现在任意位置
#
# ======================================================================================================================


class SwiC(_TypeNameUid):
    fields, line, field_index = _build([[1, 2, CARD_TYPE, 'A2', 'C '],
                                        [3, 80, '注释', 'A78']])


# 可能考虑不用这种方式进行注释
class BpaA1(_TypeNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', '.', '注释卡类型1'],
                                        [2, 80, '注释内容', 'A79']])


class BpaA2(_TypeNameUid):
    fields, line, field_index = _build([[1, 1, CARD_TYPE, 'A1', ' ', '注释卡类型2'],
                                        [2, 80, '注释内容', 'A79']])



# ======================================================================================================================

card_types = {cls[1].fields[cls[1].field_index[CARD_TYPE]].default: cls[1]
              for cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
              if '_' not in cls[0]}

# ======================================================================================================================