import datetime

from scipy.interpolate import interp1d
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

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

forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=None, random_state=None)

forest.fit(X_train, Y_train)

df = pd.read_csv(test_dir)
# 对年龄列进行线性插值
age_notna = df['Age'].dropna()
age_interpolate = interp1d(age_notna.index, age_notna.values, kind='linear', bounds_error=False, fill_value="extrapolate")
# 调整插值范围
x_new = range(len(df))
# 插值
df['Age'] = age_interpolate(x_new)
X_test = df.iloc[:, 1:].values
X_test = np.delete(X_test, [1, 6, 8, 9], 1)
print(df.isna().sum())
X_test[(X_test == "male")] = 0
X_test[(X_test == "female")] = 1
y_test_pred = forest.predict(X_test)
rdf = pd.read_csv(res_dir)

num = X_test.shape[0]

now_id = 0

for res in y_test_pred:
    rdf.at[now_id, 'Survived'] = res
    now_id = now_id + 1

time = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')

rdf.to_csv(time + '.csv', index=False)