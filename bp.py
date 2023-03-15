# 导入相关的库
import datetime

import paddle
import numpy as np
from data_process import get_TITANIC_dataloader

train_loader, test_loader = get_TITANIC_dataloader()

# 定义模型结构
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D, Linear


# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.in1 = Linear(in_features=6, out_features=5)
        self.in2 = Linear(in_features=5, out_features=4)
        self.in3 = Linear(in_features=4, out_features=3)
        self.fc = Linear(in_features=3, out_features=2)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs, label):
        x = self.in1(inputs)
        x = F.relu(x)
        x = self.in2(x)
        x = F.relu(x)
        x = self.in3(x)
        x = self.fc(x)
        if label is not None:
            acc = paddle.metric.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


# 在使用GPU机器时，可以将use_gpu变量设置成True
use_gpu = True


# paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')

# 仅优化算法的设置有所差别
def train(model):
    model = MNIST()
    model.train()

    # 可以选择其他优化算法的设置方案（可修改）
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    # opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

    # 训练epoch（可修改）
    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据
            xData = [t[0] for t in data]
            labels = [t[1] for t in data]
            labels = np.reshape(labels, (len(labels), 1))
            xData = paddle.to_tensor(xData, dtype="float32")
            labels = paddle.to_tensor(labels, dtype="int32")

            # 前向计算的过程
            predicts, acc = model(xData, labels)

            # 计算损失，取一个批次样本损失的平均值（可修改）
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 100 == 0:
                print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
                                                                            acc.numpy()))

            # 后向传播，更新参数，消除梯度的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'Titanic.pdparams')


# 创建模型
model = MNIST()
# 启动训练过程
train(model)

import pandas as pd

df = pd.read_csv("./work/gender_submission.csv")


def evaluation(model):
    print('start evaluation .......')
    # 定义预测过程
    params_file_path = 'Titanic.pdparams'
    # 加载模型参数
    param_dict = paddle.load(params_file_path)
    model.load_dict(param_dict)

    model.eval()
    eval_loader = test_loader

    now_id = 0

    for batch_id, data in enumerate(eval_loader()):
        images = data
        images = paddle.to_tensor(images)
        predicts = model(images, None)
        predicts = np.argmax(predicts, axis=1)
        for res in predicts:
            df.at[now_id, 'Survived'] = res
            now_id = now_id + 1

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
    df.to_csv(time + '.csv', index=False)


model = MNIST()
evaluation(model)
