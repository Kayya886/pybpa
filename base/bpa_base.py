# -*- coding:utf-8 -*-
"""
# Name:         transient_stability: bpa_base
# Author:       MilkyDesk
# Date:         2021/6/27 11:05
# Description:
#   
"""

from base.bpa_str import FFFloat2Str, BPA_LINE_LEN
from fortranformat import FortranRecordReader, FortranRecordWriter

# ------------------------- fields --------------------------

CARD_TYPE = 'TYPE'
# ------------------------- fields --------------------------


def _build(param_list):
    """
    根据_Field的初始值列表，补全空格至完整的行.
    l = [' ' or int]: 空格表示填充，int表示【下标】，具体为f[int]
    f = [_Field]: l(或_Card)涉及到的_Field
    d = {_Field.name: int}: 可以根据_Field.name找到【下标】
    @param param_list:
    @return: f: field_list, l: bline, d: {field_name: field}
    """
    f, l, index = [], [], {}

    p = 0
    """p是即将使用的bpa行下标"""
    for i, v in enumerate(param_list):
        _f = _Field(*v)
        f.append(_f)
        assert p <= _f.st
        if _f.st == p:
            l.append(i)
        else:
            l.append(' ' * (_f.st - p))
            l.append(i)
        p = _f.ed
        index.update({_f.name: i})
    if p < BPA_LINE_LEN:
        l.append(' ' * (BPA_LINE_LEN - p))

    return f, l, index


class _Field:

    reader_dict = {}
    writer_dict = {}

    def __init__(self, st, ed, name, format, default=0, help=None):
        """
        初始化顺序以bpa阅读顺序为准，方便录入：
        起止列，说明，格式，默认值，帮助
        """
        assert st <= ed
        self.st = st - 1  # bpa 1开始，python 0开始
        self.ed = ed  # 索引时不会索引到ed项
        self.name = name
        self.format = format
        self.default = default
        self.help = help
        self.reader = _Field._assign_reader(format)
        self.writer = _Field._assign_writer(format)
        self.width = int(self.format[1:].split('.')[0])
        assert self.ed - self.st == self.width, 'st,ed计算宽度与format不符！'

    def read(self, s: str):
        return self.reader.read(s)[0]

    def write(self, value):
        """保证输出的字符串被bpa识别时宽度一致"""
        if 'A' == self.format[0]:
            if len(value) != self.width:
                value = value.ljust(self.width)
            return bytes(self.writer.write([value]), encoding='gbk')[:self.width].decode(encoding='gbk', errors='ignore')
        else:
            return self.writer.write([value])

    @staticmethod
    def _assign_reader(format: str):
        if format in _Field.reader_dict:
            return _Field.reader_dict.get(format)
        else:
            r = FortranRecordReader(format)
            _Field.reader_dict.update({format: r})
            return r

    @staticmethod
    def _assign_writer(format: str):
        if format in _Field.writer_dict:
            return _Field.writer_dict.get(format)
        else:
            a = format[1:].split('.')
            if format[0] == 'F':  # and (int(a[0]) <= int(a[1]) + 1 or int(a[1]) == 0):  # ffWriter，当Fa.b，a<b+1时报错
                r = FFFloat2Str(int(a[0]), int(a[1]))
            else:
                r = FortranRecordWriter(format)
            _Field.writer_dict.update({format: r})
            return r


    # def str(self, bline: str):
    #     return bpa_str2x(bline[self.st, self.ed], t=self.t, f=self.format, default=self.default)


class _Card:
    """
    定位：
    提供_Feild的组织和管理工具
    行为：
    1. 初始化: 将规则的bpa行转换成结构化的行{space or _Feild}, 结构位于__class__.bline，值位于self.values
    2. __str__: 将结构化的行转换成字符串bpa行
    3. get/set feild values: 通过域名称访问和修改值，或者将值转换成域格式，
        eg:
        self.values  = self.field[i].read(str)
        str         = self.field[i].write(self.values[i])
    4. 实现类包含了：
        fields: list,
        bline: list of field or str,
        field_index: dict {field_name: field_index}
        values: list
    """

    def __init__(self, bline: str):
        """将str行结构化为结构化的行。以前的bpa_card_line()转换是在这里，但是考虑到喂进来的数据本来就应该是好的，于是去掉这个。"""
        self.values = [f.read(bline[f.st: f.ed].decode('gbk', errors='ignore'))
                       if f.read(bline[f.st: f.ed].decode('gbk', errors='ignore')) else f.default
                       for f in self.__class__.fields]
        self.type = self.fields[self.field_index[CARD_TYPE]].default

    def __str__(self):
        """将结构化的card还原回str行"""
        return ''.join([x if type(x) is str else
                        (self.fields[x].write(self.values[x]) if self.values[x] else
                         ' ' * self.fields[x].width)
                        for x in self.__class__.line]) + '\n'

    def __eq__(self, other):
        try:
            return self.name == other.name
        except:
            pass
        return False

    def get_value(self, field_name):
        if field_name in self.field_index:
            return self.values[self.field_index[field_name]]
        raise KeyError('field_name not in self.field_index')

    def get_field(self, field_name):
        if field_name in self.field_index:
            return self.fields[self.field_index[field_name]]
        raise KeyError('field_name not in self.field_index')


def _to_card(bline: str, card_dict: dict):
    # 10: max(len(card_dict.keys))
    l = bline.decode('gbk', errors='ignore')
    for i in range(10, -1, -1):
        key = l[:i]
        if key in card_dict:
            c = None
            try:
                c = card_dict[key](bline)
            finally:
                return c
    # raise TypeError('cannot found a specific card!')
    return None


