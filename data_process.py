import numpy as np
import pandas as pd
from paddle import fluid
from scipy.interpolate import interp1d

train_dir = "./work/train.csv"
test_dir = "./work/test.csv"
res_dir = "./work/gender_submission.csv"

df = pd.read_csv(train_dir)
# 对年龄列进行线性插值
age_notna = df['Age'].dropna()
age_interpolate = interp1d(age_notna.index, age_notna.values, kind='linear', bounds_error=False, fill_value="extrapolate")
# 调整插值范围
x_new = range(len(df))
# 插值
df['Age'] = age_interpolate(x_new)
X_train = df.iloc[:, 2:].values
X_train = np.delete(X_train, [1, 6, 8, 9], 1)
Y_train = df.iloc[:, 1].values
X_train[(X_train == "male")] = 0
X_train[(X_train == "female")] = 1
train_num = X_train.shape[0]

df = pd.read_csv(test_dir)
# 对年龄列进行线性插值
age_notna = df['Age'].dropna()
age_interpolate = interp1d(age_notna.index, age_notna.values, kind='linear', bounds_error=False, fill_value="extrapolate")
# 调整插值范围
x_new = range(len(df))
# 插值
df['Age'] = age_interpolate(x_new)
print(df['Age'].describe())
X_test = df.iloc[:, 1:].values
X_test = np.delete(X_test, [1, 6, 8, 9], 1)
X_test[(X_test == "male")] = 0
X_test[(X_test == "female")] = 1
test_num = X_test.shape[0]


def train_reader():
    for i in range(train_num):
        X_train[i][5] = X_train[i][5] / 513
        X_train[i][2] = X_train[i][5] / 80
        x = X_train[i].astype(dtype='float32')
        y = Y_train[i]
        yield x, y


def test_reader():
    for i in range(test_num):
        X_test[i][5] = X_test[i][5] / 513
        X_test[i][2] = X_test[i][2] / 80
        x = X_test[i].astype(dtype='float32')
        yield x


TrainSet_reader = fluid.io.batch(train_reader, batch_size=5)
TestSet_reader = fluid.io.batch(test_reader, batch_size=5)


def get_TITANIC_dataloader():
    return TrainSet_reader, TestSet_reader
