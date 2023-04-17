import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=16)
        # self.pool3 = nn.AvgPool1d(kernel_size=4)
        self.fc1 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64)
        x = self.fc1(x)
        return x

# 实例化模型和损失函数
model = Net()
criterion = nn.MSELoss()


from nirs.parameters import X_train_copy,y_train_copy,X_test,y_test
# 读入数据并转换为PyTorch Tensor格式
x =X_train_copy
y = y_train_copy
x_tensor = torch.Tensor(x.reshape(-1, 1, 256))
y_tensor = torch.Tensor(y.reshape(-1, 1))

cur_lr = 0.1
# 定义学习率衰减率和衰减周期
decay_rate = 0.97
decay_steps = 200
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=cur_lr)
num_epochs=20000
# 训练模型
for epoch in range(num_epochs):
    # 前向传播
    lr = cur_lr * decay_rate ** (epoch // decay_steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练过程中的损失值
    if (epoch+1) % 100 == 0:
        print(lr)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 预测并输出结果
with torch.no_grad():
    X_test = torch.Tensor(X_test.reshape(-1, 1, 256))
    outputs = model(X_test)
    outputs  =outputs.numpy()
    print( r2_score(y_test, outputs))
    print(np.sqrt(mean_squared_error(y_test/100, outputs/100)))

    X_test= X_train_copy
    y_test = y_tensor
    outputs = model(x_tensor)
    outputs  =outputs.numpy()
    print( r2_score(y_test, outputs))
    print(np.sqrt(mean_squared_error(y_test/100, outputs/100)))
