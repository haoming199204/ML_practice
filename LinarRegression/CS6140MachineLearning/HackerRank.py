# coding:utf-8
# demo
from click._compat import raw_input

class Data:
    def __init__(self):
        self.x = []
        self.y = 0.0


def WX(d, w):
    ans = 0.0
    for i in range(0, len(w)):
        ans += w[i] * d.x[i]
    return ans


def Gradient(d, w, alpha):
    for i in range(0, len(w)):
        tmp = 0.0
        for j in range(0, len(d)):
            tmp += alpha * d[j].x[i] * (WX(d[j], w) - d[j].y)
        w[i] -= tmp


def getValues(d, w):
    res = 0.0
    for i in range(0, len(d)):
        tmp = WX(d[i], w)
        res += (d[i].y - tmp) * (d[i].y - tmp)
    return res


def Iterator(d, w):
    alpha = 0.005
    delta = 0.000001
    oldVal = getValues(d, w)
    Gradient(d, w, alpha)
    newVal = getValues(d, w)
    while abs(oldVal - newVal) > delta:
        oldVal = newVal
        Gradient(d, w, alpha)
        newVal = getValues(d, w)

def main():
    while True:
        try:
            d = []
            w = []
            F, N = map(int, raw_input().split())
            for i in range(0, N):
                t = Data()
                t.x = map(float, raw_input().split())
                t.x.insert(0, 1.0)
                t.y = t.x.pop()
                d.append(t)
            for i in range(0, F + 1):
                w.append(0)
            Iterator(d, w)
            N = int(raw_input())
            for i in range(0, N):
                t = Data()
                t.x = map(float, raw_input().split())
                t.x.insert(0, 1.0)
                print('%.2f' % WX(t, w))
        except EOFError:
            break

if __name__ == '__main__':
    main()
