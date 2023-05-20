import pathlib
import numpy as np
from numpy import mat, zeros

from nirs.nirs_processing import piecewise_polyfit_baseline_correction

def kennardstonealgorithm(x_variables, k_rate):
    k = int(len(x_variables) * k_rate)
    x_variables = np.array(x_variables)
    original_x = x_variables
    distance_to_average = ((x_variables - np.tile(x_variables.mean(axis=0), (x_variables.shape[0], 1))) ** 2).sum(
        axis=1)
    max_distance_sample_number = np.where(distance_to_average == np.max(distance_to_average))
    max_distance_sample_number = max_distance_sample_number[0][0]
    selected_sample_numbers = list()
    selected_sample_numbers.append(max_distance_sample_number)
    remaining_sample_numbers = np.arange(0, x_variables.shape[0], 1)
    x_variables = np.delete(x_variables, selected_sample_numbers, 0)
    remaining_sample_numbers = np.delete(remaining_sample_numbers, selected_sample_numbers, 0)
    for iteration in range(1, k):
        selected_samples = original_x[selected_sample_numbers, :]
        min_distance_to_selected_samples = list()
        for min_distance_calculation_number in range(0, x_variables.shape[0]):
            distance_to_selected_samples = ((selected_samples - np.tile(x_variables[min_distance_calculation_number, :],
                                                                        (selected_samples.shape[0], 1))) ** 2).sum(
                axis=1)
            min_distance_to_selected_samples.append(np.min(distance_to_selected_samples))
        max_distance_sample_number = np.where(
            min_distance_to_selected_samples == np.max(min_distance_to_selected_samples))
        max_distance_sample_number = max_distance_sample_number[0][0]
        selected_sample_numbers.append(remaining_sample_numbers[max_distance_sample_number])
        x_variables = np.delete(x_variables, max_distance_sample_number, 0)
        remaining_sample_numbers = np.delete(remaining_sample_numbers, max_distance_sample_number, 0)

    return selected_sample_numbers, remaining_sample_numbers

import numpy as np

def spxy(X, y, test_size=0.2):
    M = X.shape[0]  # Number of objects
    Ncal = round((1 - test_size) * M)

    dminmax = np.zeros(Ncal)  # Initializes the vector of minimum distances.
    samples = np.arange(M)
    Dx = np.zeros((M, M))  # Inicializes the matrix of X-distances.
    Dy = np.zeros((M, M))  # Inicializes the matriz de y-distances.
    for i in range(M-1):
        xa = X[i, :]
        ya = y[i]
        for j in range(i + 1, M):
            xb = X[j, :]
            yb = y[j]
            Dx[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)
    Dxmax = np.max(np.max(Dx))
    Dymax = np.max(np.max(Dy))
    D = Dx / Dxmax + Dy / Dymax  # Combines the X and y distances.
    # D is an upper triangular matrix.
    # D[i,j] is the distance between objects i and j (j > i).
    maxD = np.max(D, axis=0)
    index_row = np.argmax(D, axis=0)
    # maxD is a row vector containing the largest element for each column of D.
    # index_row is the row in which the largest element of the column if found.
    index_column = np.argmax(maxD)
    # index_column is the column containing the largest element of matrix D.
    m = np.zeros(Ncal, dtype=int)
    m[0] = index_row[index_column]
    m[1] = index_column
    for i in range(2, Ncal):
        pool = np.setdiff1d(samples, m[:i])
        # Pool is the index set of the samples that have not been selected yet.
        dmin = np.zeros(M - i)
        # dmin will store the minimum distance of each sample in “pool” with respect to the previously selected samples.
        for j in range(M - i):
            indexa = pool[j]
            d = np.zeros(i )
            for k in range(i ):
                indexb = m[k]
                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            dmin[j] = np.min(d)
        # At each iteration, the sample with the largest dmin value is selected.
        index = np.argmax(dmin)
        m[i] = pool[index]

    a = np.delete(np.arange(M), m)
    spec_train = X[m, :]
    target_train = y[m]
    spec_test = X[a, :]
    target_test = y[a]

    return spec_train, spec_test, target_train, target_test


def spxy1(x, y, test_size=0.2):
    x_backup = x
    y_backup = y
    M = x.shape[0]
    N = round((1 - test_size) * M)
    samples = np.arange(M)

    y = (y - np.mean(y)) / np.std(y)
    D = np.zeros((M, M))
    Dy = np.zeros((M, M))

    for i in range(M - 1):
        xa = x[i, :]
        ya = y[i]
        for j in range((i + 1), M):
            xb = x[j, :]
            yb = y[j]
            D[i, j] = np.linalg.norm(xa - xb)
            Dy[i, j] = np.linalg.norm(ya - yb)

    Dmax = np.max(D)
    Dymax = np.max(Dy)
    D = D / Dmax + Dy / Dymax

    maxD = D.max(axis=0)
    index_row = D.argmax(axis=0)  # 返回axis轴方向最大值的索引
    index_column = maxD.argmax()

    m = np.zeros(N)
    m[0] = index_row[index_column]
    m[1] = index_column
    m = m.astype(int)

    dminmax = np.zeros(N)
    dminmax[1] = D[m[0], m[1]]

    for i in range(2, N):
        pool = np.delete(samples, m[:i])
        dmin = np.zeros(M - i)
        for j in range(M - i):
            # pool是还未被选进去的  和选中的距离
            indexa = pool[j]
            d = np.zeros(i)
            for k in range(i):
                indexb = m[k]

                # 半角矩阵

                if indexa < indexb:
                    d[k] = D[indexa, indexb]
                else:
                    d[k] = D[indexb, indexa]
            #dmin 未选取的样本  到选取样本的最小距离
            dmin[j] = np.min(d)
        # 最小距离的最大值
        dminmax[i] = np.max(dmin)
        index = np.argmax(dmin)
        m[i] = pool[index]

    m_complement = np.delete(np.arange(x.shape[0]), m)

    spec_train = x[m, :]
    target_train = y_backup[m]
    spec_test = x[m_complement, :]
    target_test = y_backup[m_complement]

    return spec_train, spec_test, target_train, target_test



def loadDataSet01(filename, Separator=', '):
    import pathlib
    fr = open(filename)
    arrayLines = fr.readlines()
    assert len(arrayLines) != 0
    num = len(arrayLines[0].split(Separator)) - 1
    row = len(arrayLines)
    x = mat(zeros((row, num)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split(Separator)
        curLine = [float(i) for i in curLine]
        x[index, :] = curLine[0: -1]
        # y[index, :] = curLine[-1]/100
        y[index, :] = curLine[-1]
        index += 1
    fr.close()
    return np.array(x), np.array(y).ravel()


xpoints = np.array(
    [919.8078, 923.907776, 927.984375, 932.037781, 936.068176, 940.075684, 944.060486, 948.022827, 951.96283,
     955.880676, 959.776611, 963.650696, 967.503235, 971.33429, 975.144104, 978.9328, 982.700623, 986.447754,
     990.174316, 993.880432, 997.566467, 1001.232483, 1004.878601, 1008.505066, 1012.112122, 1015.699829,
     1019.268433, 1022.817993, 1026.348755, 1029.861206, 1033.354858, 1036.830566, 1040.288208, 1043.727661,
     1047.14978, 1050.554077, 1053.94104, 1057.311157, 1060.66394, 1063.999878, 1067.319458, 1070.622314,
     1073.908936, 1077.179688, 1080.434204, 1083.673096, 1086.896362, 1090.104248, 1093.296875, 1096.474609,
     1099.637207, 1102.7854, 1105.918945, 1109.038086, 1112.143188, 1115.234253, 1118.311523, 1121.375122,
     1124.425293, 1127.462402, 1130.486206, 1133.496948, 1136.495239, 1139.480713, 1142.453979, 1145.415161,
     1148.364014, 1151.301147, 1154.226685, 1157.140503, 1160.043213, 1162.934692, 1165.815308, 1168.684937,
     1171.544189, 1174.392822, 1177.231323, 1180.059692, 1182.878052, 1185.686768, 1188.486084, 1191.275879,
     1194.056519, 1196.828125, 1199.590942, 1202.345093, 1205.09082, 1207.828247, 1210.557373, 1213.278687,
     1215.99231, 1218.69812, 1221.396606, 1224.087891, 1226.771973, 1229.449341, 1232.120117, 1234.784058,
     1237.441772, 1240.093384, 1242.738892, 1245.37854, 1248.012817, 1250.641357, 1253.264648, 1255.883057,
     1258.496216, 1261.104736, 1263.708862, 1266.30835, 1268.903687, 1271.494995, 1274.082397, 1276.66626,
     1279.24646, 1281.823364, 1284.397095, 1286.967773, 1289.535889, 1292.101196, 1294.664062, 1297.224731,
     1299.783325, 1302.339722, 1304.894775, 1307.447998, 1310, 1312.550659, 1315.100342, 1317.649292,
     1320.197388, 1322.745117, 1325.292358, 1327.839722, 1330.386841, 1332.934326, 1335.4823, 1338.030762,
     1340.579834, 1343.130005, 1345.681396, 1348.233643, 1350.787598, 1353.343262, 1355.900391, 1358.459961,
     1361.021606, 1363.585449, 1366.151611, 1368.720947, 1371.292725, 1373.867676, 1376.446045, 1379.02771,
     1381.612915, 1384.202026, 1386.794922, 1389.391968, 1391.993408, 1394.599243, 1397.209717, 1399.825195,
     1402.445312, 1405.070801, 1407.70166, 1410.338135, 1412.980103, 1415.628296, 1418.282227, 1420.942627,
     1423.609375, 1426.282837, 1428.962769, 1431.649902, 1434.344238, 1437.045532, 1439.754761, 1442.471558,
     1445.196045, 1447.928345, 1450.669312, 1453.418335, 1456.176025, 1458.942505, 1461.717773, 1464.502197,
     1467.295776, 1470.098877, 1472.911743, 1475.734253, 1478.56665, 1481.409302, 1484.262329, 1487.125732,
     1489.999878, 1492.88501, 1495.781006, 1498.688232, 1501.607056, 1504.537354, 1507.479248, 1510.433105,
     1513.39917, 1516.377441, 1519.368164, 1522.371704, 1525.387695, 1528.417114, 1531.459473, 1534.515137,
     1537.584229, 1540.667236, 1543.764038, 1546.874878, 1550, 1553.139648, 1556.293457, 1559.462402,
     1562.64624, 1565.845093, 1569.059326, 1572.289062, 1575.53418, 1578.795654, 1582.072632, 1585.365967,
     1588.675659, 1592.001953, 1595.344849, 1598.704834, 1602.081787, 1605.47583, 1608.887573, 1612.316772,
     1615.763672, 1619.228638, 1622.711914, 1626.213013, 1629.73291, 1633.271729, 1636.828979, 1640.405273,
     1644.000977, 1647.616089, 1651.25061, 1654.905029, 1658.578979, 1662.273193, 1665.988037, 1669.722778,
     1673.478271, 1677.254883, 1681.052368, 1684.870728, 1688.710693, 1692.572144], dtype=int)

import win32com.client as win32

def convert_png_to_wmf(png_path, wmf_path):
    # 创建Word应用程序对象
    word_app = win32.gencache.EnsureDispatch("Word.Application")

    # 打开一个新的Word文档
    doc = word_app.Documents.Add()

    # 在文档中插入PNG图像
    doc.InlineShapes.AddPicture(png_path)

    # 将文档保存为WMF格式


    doc.SaveAs(wmf_path, FileFormat=3)

    # 关闭Word应用程序
    word_app.Quit()

def transWMF():
    import glob

    # 获取当前文件夹下所有PNG图片的文件路径
    png_files = glob.glob("*.png")

    # 打印每个PNG图片的文件路径
    for file_path in png_files:
        convert_png_to_wmf(file_path, file_path[:-3]+"WMF")
def saveWMF(filename):

    import matplotlib.pyplot as plt
    from PIL import Image
    import win32com.client

    import os

    script_path = os.path.abspath(__file__)
    script_directory = os.path.dirname(script_path)
    # 保存为PNG格式
    tmp = f"figure111.png"
    plt.savefig(tmp)

    # 使用Pillow库打开PNG图像
    image = Image.open(tmp)
    filename =  f"{filename}"
    import time

    convert_png_to_wmf(tmp,filename)

    # 删除中间的PNG文件
    image.close()
    print(f"save in {filename}")
    import os


    os.remove(tmp)


class Parameter:
    def __int__(self, optimal=False):
        self.optimal = optimal


para = Parameter()

# 预处理的算法的参数都是固定的
preprocess_args = {
    # "method":["SG","DT"]
    "method": ["SG"]

}

# 特征算法的参数设置
feature_selection_args = {

    "method": ["cars", "pca"],
    "pca": {
        # 主成分个数
        "n_components": 56,
    },
    "plsr": {
        "n_components": 11,
    },
    "cars": {
        # cars的index
        "index": [3, 5, 8, 14, 26, 31, 41, 43, 48, 49, 53, 59, 65, 72, 80, 81, 85, 87, 90, 110, 111, 116, 120, 122, 126,
                  127, 128, 153, 158, 175, 180, 184, 197, 198, 202, 205, 206, 214, 227, 240, 241, 255],
    },

    "spa": {
        # cars的index
        "index": [219,6,195,134,46,3,182,158,179,254,191,188,161,84,165,2,166,171,176,193,167,174,172,196,163,177,198,175,189,159,242,180,168],
    },

    'lasso': {

        'alpha': 0.0005,
        'normalize': False

    },
    'new':{
        # 'index':[252,13,182,46,6,134,159,184,165,199,2,171,167,179,175,203,176,169,1,170]
        'index':[1,6,28,55,57,113,126,165,167,169,177,180,181,229,233]
    },
    'yumi':{
      # 'index':[47,52,54,67,182,208,214,217,236,239,260,293,310,319,331,355,361,378,388,396,403,408,455,473,495,512,540,562,575,586,601,602,604,616,692,699]
      # 'index':[60,157,165,174,194,208,210,235,252,259,281,289,294,334,344,358,362,369,382,408,413,434,437,439,452,454,470,487,494,505,510,526,541,543,544,551,560,573,593,604,619,641,649,651,659,667,680,695,698,699]
      'index':[8,9,36,49,74,75,96,116,124,134,144,145,160,198,220,274,276,277,295,307,308,325,329,341,344,381,399,431,480,492,512,515,527,536,547,548,560,568,576,585,609,619,621,625,644,654,665,666,672,677]    },
    # 使用index的特征提取方法
    "mp5spec_moisture":{
        'index':[156, 221 ,500, 610]
    }
    ,
    "index_set": ["cars", "ga","spa",'new',"yumi",'mp5spec_moisture'],

}

# 模型的参数设置
model_args = {

    "model": "plsr",
    'lar':{

        'alpha':0.0003,
        'normalize':False

    },
    # svr

    "lssvm":{
        "C":10,
        "gamma":3.7,
        'sigma':0.1
        # 'epsilon':0.0020
    },
    "svr": {
        # "C": 100,
        # "gamma": 'scale',
        # "C": 10,
        # "gamma":23,
# 9.91963387e+02 2.52168033e-01 2.39470555e-01  随机
# 7.68476853e+02 4.96147122e+00 1.64400238e-01		  AGA

        # 9.73668478e+02 3.71832582e+00 2.08149113e-03  IPSO
        # 641.82456427   3.56104933   0.7002129
        # "C":577.2508287,
        # "gamma": 65.02340764,
        # "C": 577.2508287,
        # "gamma": 65.02340764,
         # 'epsilon': 0.7002129	 ,
# C=768.47, gamma=4.9614, epsilo=0.1644
# C=641.82, gamma=3.5610, epsilo=0.7002
# C=973.66, gamma=3.7183, epsilo=0.0020
# 'C': 998.8046, 'gamma': 0.4724, 'epsilon': 0.7003, 'kernel': 'rbf'
# 'C': 768.47, 'gamma':4.9614, 'epsilon':0.1644, 'kernel': 'rbf'
# 'C': 973.66, 'gamma':3.7183,  'epsilon':0.0020, 'kernel': 'rbf'
#         C=991.96, gamma=0.2521, epsilo=0.2394
# 'C':991.96, 'gamma':.2521,  'epsilon':0.2394, 'kernel': 'rbf'
#         C=128, gamma=16, epsilo=0.297
# 'C':128, 'gamma':16,  'epsilon':0.297, 'kernel': 'rbf'

#         C=577.25, gamma=65.023
# 'C': 1906.4584194530692, 'gamma': 477.71211608797796, 'epsilon': 0.0020853886390257348, 'kernel': 'rbf'
# 'C':400.25, 'gamma':25.023
#         'kernel':'linear'
        'C': 10, 'gamma': 1, 'epsilon': 0.1, 'kernel': 'rbf'
        # "C": 100,
        # "kernel": "linear",
    },
    'rf':{
'n_estimators':38, 'max_depth':12,  'min_samples_split': 2, 'min_samples_leaf': 2,'max_features':5

    #     criterion = "mse", min_samples_split = 2, min_samples_leaf = 2, max_depth = 12, max_features = min(5, len(
    # x_train[0])), bootstrap = False
    },

    'adaboost':{

    },

    # pls
    "plsr": {
        "n_components": 11,
    },
    "bpnn": {

        "hidden_size": 100,
        # "activation": 'logistic',
        # "solver": 'sgd',
        # "alpha": 0.0001,
        "learning_rate": 0.01,
        # "learning_rate_init": 0.01,
        # "power_t": 0.5,
        "num_epochs": 10000,
        # "shuffle": False,
        # "tol": 0.0001,
        # "verbose": False,
        # "warm_start": False,
        # "momentum": 0.9, "nesterovs_momentum": True, "early_stopping": False,
        # "validation_fraction": 0.1, "beta_1": 0.937, "beta_2": 0.999,
        # "epsilon": 1e-08,
        #
        # "n_iter_no_change": 10,
        # "max_fun": 15000

    },
    # SG+GT
    # "rbf":{'hidden_shape': 250, 'sigma': 22.19206269244502, 'alpha': 1480.8036479688653, 'kernel': 'gaussian', 'gamma': 0.07744522935425945},
    # SNV
    # "rbf": {'hidden_shape': 238, 'sigma': 210.55817241205375, 'alpha': 384.21457791907585, 'kernel': 'poly', 'gamma': 0.2616510138531062},
    #     DT
    "rbf": {'hidden_shape': 255, 'sigma': 20.48862236375201, 'alpha': 128.6355374786053, 'kernel': 'gaussian',
            'gamma': 0.8264571838644237}

}

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans




#model: SG_NONE_LINEAR  DT_NONE_LINEAR
# R2: 0.9994
# RMSECV: 0.0001
# r2: 0.9996
# RMSEP: 0.0000
# RPD: 50.8678

# MAE: 0.0000

import pathlib

dir = pathlib.Path("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/corn/")

data={}
for i in dir.glob("*.txt"):

    abs = str(i.absolute())
    name = i.name[:i.name.rfind(".")]

    x,y = loadDataSet01(abs,",")
    data[name] = spxy1(x,y,test_size=0.3)




# m5spec_moisture, m5spec_moisture_y = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/corn/m5spec_moisture.txt".replace("/","\\"),Separator=",")



# m5spec_moisture, m5spec_moisture_y = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/corn/mp5spec_moisture.txt".replace("/","\\"),Separator=",")
# m5spec_moisture, m5spec_moisture_y = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/corn/mp6spec_moisture.txt".replace("/","\\"),Separator=",")



# m5spec_moisture, m5spec_moisture_y = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/corn/mp5spec_oil.txt".replace("/","\\"),Separator=",")

# selected_sample_numbers, remaining_sample_numbers = kennardstonealgorithm(m5spec_moisture, 0.8)
# m5spec_moisture_train, m5spec_moisture_test = m5spec_moisture[selected_sample_numbers], m5spec_moisture[
#     remaining_sample_numbers]
# m5spec_moisture_y_train, m5spec_moisture_y_test = m5spec_moisture_y[selected_sample_numbers], m5spec_moisture_y[
#     remaining_sample_numbers]


# m5spec_moisture_train, m5spec_moisture_test, m5spec_moisture_y_train, m5spec_moisture_y_test= spxy1(m5spec_moisture,m5spec_moisture_y)


def print_min_max(a:np.array):
    print(f"{np.min(a)} --- {np.max(a)}")

data_indice = ['m5spec_moisture','m5spec_oil','m5spec_protein','m5spec_starch',
               'mp5spec_moisture','mp5spec_oil','mp5spec_protein','mp5spec_starch','mp6spec_moisture','mp6spec_oil',
               'mp6spec_protein','mp6spec_starch']

indice = 8
X_train,X_test,y_train,y_test =data.get(data_indice[indice])
# X_train = piecewise_polyfit_baseline_correction(X_train)
# X_test = piecewise_polyfit_baseline_correction(X_test)
# print_min_max(X_train)
# print_min_max(X_test)

para.path=data_indice[indice] + ".xlsx"




X_test, y_test = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/test.txt".replace("/","\\"))
X_train, y_train = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/train.txt".replace("/","\\"))
X_train = np.log10(1/X_train)
X_test = np.log10(1/X_test)


X_train_copy, y_train_copy = loadDataSet01("C:/Users/Administrator/PycharmProjects/nirs_water_prediction/data/train_copy.txt".replace("/","\\"))
X_train_copy = np.log10(1/X_train_copy)

print_min_max(y_train)
print_min_max(y_test)