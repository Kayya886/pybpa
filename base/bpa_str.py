#!/usr/bin/python
# -*- coding: gbk -*-


BPA_LINE_LEN = 80

def bpa_str2float(s: str, f: int = None, default: float = 0):
    """
    F6.5����ʾҪ������һ����������ռ�е��������Ϊ6�У�ȱʡС������λ��Ϊ5λ��
    ������롮123456��������û��С���㣬�����ȱʡ����Ϊ1.23456��������롮10������������ȱʡ����ǰ���Զ����㣬С������ȡ5λ�����������0.00010��
    ������롮10.����������С���㣬��������Ϊ10.0�������д��.12345�������������Ϊ0.12345��
    �����д�ĸ�����û��ռ��ָ�����У�����ڸ�ʽF6.5���롮10.����3�У�����Կո���������пո�Ĵ�����һ�¡�
    @param default:
    @param f:
    @param s:
    @return:
    """
    length = len(s)
    s = s.strip()
    if s:
        return (float(s + '0')) if '.' in s else (float(s) / (10 ** f))
    else:
        if default is None:
            raise ValueError('���ڷ��գ�')
        return default


def bpa_str(s: str):
    return s.strip()


def bpa_str2int(s: str, default: int = None):
    s = s.strip()
    if len(s) > 0:
        return int(s)
    else:
        if default is None:
            raise ValueError('���ڷ��գ�')
        return default


def bpa_str2x(s: str, t: type, f: int = None, default = None):
    if t == str:
        return bpa_str(s)
    elif t == int:
        return bpa_str2int(s, default=default)
    elif t == float:
        return bpa_str2float(s, f, default)


def bpa_card_line(bline: str):
    """
    �淶��bpa�ļ��С�
    ���ڲ��Ǵ����е��У���������ļ��в��뵽BPA_LINE_LEN���ȣ�ȥ�����з�
    """
    if bline == b'\r\n' or bline == b'\n':
        return b'\n'
    elif bline[-2:] == b'\r\n':
        return bline[:-2].ljust(BPA_LINE_LEN)
    elif bline[-1:] == b'\n':
        return bline[:-1].ljust(BPA_LINE_LEN)
    else:
        return bline.ljust(BPA_LINE_LEN)
    # # ������(END)
    # raise ValueError('�����˲���\\n,\\r\\n��β���У�')


class FFFloat2Str:
    def __init__(self, w, d):
        self.w = w
        self.d = d
        self.max = 10 ** (w - (1 if d else 0))
        self.min = -1 * 10 ** (w - 1 - (1 if d else 0))
        self.pfull = 10 ** (w - d - 1)
        self.nfull = -1 * 10 ** (w - d - 2)

    def write(self, fl):
        f = fl[0]
        if not f:
            return ' ' * self.w
        assert self.min < f < self.max

        f += 0.000000000001
        if self.pfull < f < self.pfull * 10 or 10 * self.nfull < f < self.nfull:
            f *= 10 ** (3 * self.d)
            return ('{:.' + str(self.d) + 'f}').format(f)[:self.w].ljust(self.w)

        s = ('{:.' + str(self.d) + 'f}').format(abs(f))
        if s[:2] == '0.':
            s = s[1:]
        if f < 0:
            s = '-' + s
        return s[:self.w].ljust(self.w)


if __name__ == '__main__':
    import fortranformat as ff
    class t:
        def __init__(self, w, b):
            self.a = ff.FortranRecordReader('F' + str(w) + '.' + str(b))
            self.b = FFFloat2Str(w, b)
        def w(self, f):
            h = self.b.write([f])
            g = self.a.read(h)[0]
            return g == f, '\t'.join([str(f), h, str(g)])

    a = FFFloat2Str(6, 5)  # todo F6.5  ��.00008,д����8.0000
    print(a.write([.00008]))
    print(a.write([-321.12345]))
    print(a.write([.12345]))
    print(a.write([-.12345]))
    #
    # b = t(5, 4)
    # for i in range(1000000):
    #     f, s = b.w(i / 10000)
    #     if not f:
    #         print(i, '\t' + s)
