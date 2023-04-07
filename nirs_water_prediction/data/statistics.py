
from PLS.utils import loadDataSet01
import numpy as np

# 统计所有的数据
# (水分含量范围)   水分含量平均值  水分含量方差
# 10 15 10.0 0.0
# 30 50 40.0 6.29
# 50 65 59.43 3.96
# 65 70 66.5 0.5
# 70 75 71.69 1.2

# x0, y0 = loadDataSet01('total.txt', ', ')
# x0, y0 = loadDataSet01('train_pre.txt', ', ')
# x0, y0 = loadDataSet01('test.txt', ', ')


# idx = [0,1,2,3,4]
#
# start = [10,30,50,65,70]
# end = [15,50,65,70,75]

def main(file):
    x0, y0 = loadDataSet01(file, ', ')
    idx=[0,1,2,3,4,5]
    start=[10,30,40,50,60,70]
    end=[19,40,50,60,70,79]

    sum = [[] for i in range(len(idx))]

    for y in y0:
        for i in idx:
            if y>= start[i] and y < end[i]:
                sum[i].append(y)
                break

    for (i,y) in enumerate(np.array(sum)):
        print(start[i],end[i], np.around(np.mean(y),2), np.around(np.std(y),2),min(y),max(y),len(y))


if __name__ == '__main__':
    main('total.txt')
    main('train_pre.txt')
    main('test.txt')