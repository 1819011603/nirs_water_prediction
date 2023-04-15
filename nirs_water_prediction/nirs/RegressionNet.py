import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.base import BaseEstimator, RegressorMixin


# 模型 必须
class Regression(nn.Module,BaseEstimator,RegressorMixin):
    def __init__(self,**params):
        '''
       super(Regression,self).__init__()
        u = min(32, max(int(c / 3),1))
        v = min(u, min(11, max(int(u/3)+1,1)))
        print("c:{}, u:{}, v:{}".format(c,u,v))
        self.l = nn.Sequential(

            nn.Linear(c,u,dtype=torch.float64), # 35 12  91.97   32 11
            nn.Sigmoid(),

            nn.Linear(u,v,dtype=torch.float64),
            nn.Sigmoid(),
            # nn.Conv1d(c, c, 5, 1, 2),
            nn.Linear(v,1,dtype=torch.float64)
        )
        '''
        super(Regression,self).__init__()
        self.params = params


        # u = min(32, max(int(c / 3),1))
        # v = min(u, min(11, max(int(u/3)+1,1)))

        self.j = 0
    def forward(self,x):
        x = torch.tensor(x)
        # x = torch.unsqueeze(x,dim=2)
        if  self.u != self.v:
            return self.l3(self.s2(self.l2(self.s1(self.l1(x)))))
        else:
            return self.l3(self.s1(self.l1(x)))


    def fit(self,X,y):
        c = len(X[0])
        u = 15
        v = 15
        self.u = u
        self.v = v
        # print("c:{}, u:{}, v:{}".format(c,u,v))

        # https://blog.csdn.net/lengyoumo/article/details/102969538
        # https://blog.csdn.net/qq_41385229/article/details/110456388

        if u != v:
            self.l1 = nn.Linear(c, u, dtype=torch.float64)  # 35 12  91.97   32 11
            self.s1 = nn.Sigmoid()
            self.l2 = nn.Linear(u, v, dtype=torch.float64)
            self.s2 = nn.Sigmoid()
            self.l3 = nn.Linear(v, 1, dtype=torch.float64)
        else:
            self.l1 = nn.Linear(c, u, dtype=torch.float64)  # 35 12  91.97   32 11
            self.s1 = nn.Sigmoid()

            self.l3 = nn.Linear(v, 1, dtype=torch.float64)
        x_train = torch.tensor(X, dtype=torch.float64)

        y_train = torch.tensor(y, dtype=torch.float64)



        optimizer = optim.Adam(self.parameters(), lr=0.1, betas=(0.937, 0.999))
        loss_func = torch.nn.MSELoss()

        loss = 1.0

        its = 20000

        for i in range(its):
            y_predict = self.forward(x_train)
            y_predict = y_predict.squeeze(-1)
            loss = loss_func(y_predict, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        self.j+=1
        print(self.j)


    def predict(self,X):

        with torch.no_grad():
            y_predict = self.forward(X)
        return y_predict.detach().numpy()


class Regression1(nn.Module):
    def __init__(self,c = 256):
        '''
       super(Regression,self).__init__()
        u = min(32, max(int(c / 3),1))
        v = min(u, min(11, max(int(u/3)+1,1)))
        print("c:{}, u:{}, v:{}".format(c,u,v))
        self.l = nn.Sequential(

            nn.Linear(c,u,dtype=torch.float64), # 35 12  91.97   32 11
            nn.Sigmoid(),

            nn.Linear(u,v,dtype=torch.float64),
            nn.Sigmoid(),
            # nn.Conv1d(c, c, 5, 1, 2),
            nn.Linear(v,1,dtype=torch.float64)
        )
        '''
        super(Regression1,self).__init__()
        # u = min(32, max(int(c / 3),1))
        # v = min(u, min(11, max(int(u/3)+1,1)))
        u = 32
        v = 8
        print("c:{}, u:{}, v:{}".format(c,u,v))

        #全连接神经网络（Multi-Layer Perception, MLP）或者叫多层感知机，是一种连接方式较为简单的人工神经网络结构，属于前馈神经网络的一种，只要有输入层、隐藏层和输出层构成，并且在每个隐藏层中可以有多个神经元。MLP 网络是可以应用于几乎所有任务的多功能学习方法，包括分类、回归，甚至是无监督学习。
        self.l = nn.Sequential(
            # https://blog.csdn.net/lengyoumo/article/details/102969538
            # https://blog.csdn.net/qq_41385229/article/details/110456388
            nn.Linear(c,u,dtype=torch.float64), # 35 12  91.97   32 11
            nn.Sigmoid(),

            # nn.Dropout(0.01),
            # nn.LeakyReLU(),

            nn.Linear(u,v,dtype=torch.float64),

            nn.Sigmoid(),
            # nn.Dropout(0.01),
            # nn.Conv1d(c, c, 5, 1, 2),
            nn.Linear(v,1,dtype=torch.float64)
        )
    def forward(self,x):

        # x = torch.unsqueeze(x,dim=2)
        return self.l(x)




class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.rnn = nn.Sequential(
            nn.LSTM(32,10,dtype=torch.float64)
        )
        self.l = nn.Sequential(
            nn.Linear(80, 17, dtype=torch.float64),  # 35 12  91.97   32 11
            nn.Sigmoid(),
            nn.Linear(17, 1, dtype=torch.float64)
        )
        self.h0 = torch.randn(2,416,10)
        self.c0 = torch.randn(2,416,10)


    def forward(self, x):
        x= x.reshape(-1,8,32)
        x = x.permute(1,0,2)

        y = self.rnn(x)
        x = y[0]
        x = x.permute(1,0,2).reshape(-1,80)


        return self.l(x)


class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers, batch):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers, batch_first=batch)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)

        batch_size, seq_len, hid_dim = y.shape
        y = y.reshape(-1, hid_dim)
        y = self.reg(y)
        y = y.reshape(batch_size, seq_len, -1)
        return y


class Transformer(nn.Module):
   def __init__(self):
       super(Transformer, self).__init__()

if __name__ == "__main__":
    net = Regression()
    for i,item in enumerate(net.l.named_children()):
        if i%2==1:
            print(str(item[1])[:-2])
import numpy as np


class RBFNet(object):
    def __init__(self, hidden_dim, sigma=1.0):
        self.hidden_dim = hidden_dim
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _gaussian(self, x, center):
        return np.exp(-np.power(x - center, 2).sum(axis=1) / (2 * np.power(self.sigma, 2)))

    def _design_matrix(self, X):
        random_idx = np.random.permutation(X.shape[0])[:self.hidden_dim]
        self.centers = X[random_idx]
        D = np.zeros((X.shape[0], self.hidden_dim))
        for i in range(X.shape[0]):
            D[i] = self._gaussian(X[i], self.centers)
        return D

    def fit(self, X, y):
        D = self._design_matrix(X)
        self.weights = np.linalg.solve(D.T @ D, D.T @ y)

    def predict(self, X):
        D = self._design_matrix(X)
        y_pred = D @ self.weights
        return y_pred


"""


nn.Linear(256,32),
nn.LeakyReLU(),
nn.Linear(32,11),
nn.LeakyReLU(),
nn.Linear(11,1)
R^2 96.69%
RMSE. 2.210449886150209

leakyRelu 
R^2 88.18%
RMSE. 4.464364657322802

nn.Linear(256,32),
nn.Sigmoid(),
nn.Linear(32,11),
nn.LeakyReLU(),
nn.Linear(11,1)
R^2 92.84%
RMSE. 2.949128187208566


"""
# net = Regression()
# print(net)

# optimizer=optim.SGD(net.parameters(),lr=0.001)
# loss_func = torch.nn.MSELoss()
# for i in range(200):
#     predition = net(x)
#     loss = loss_func(predition,y)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()


if __name__ == '__main__':
    r = Regression(c=256)


