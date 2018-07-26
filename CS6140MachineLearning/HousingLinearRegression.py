from numpy import *
import pandas
from numpy.linalg import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse


class Data(object):
    def __init__(self):
        train_data_url = "data/housing_train.txt"
        test_data_url = "data/housing_test.txt"
        names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT',
                 'MEDV']
        train_dataset = pandas.read_csv(train_data_url, names=names, sep='\s+')
        validator_dataset = pandas.read_csv(test_data_url, names=names, sep='\s+')
        self.X_train = train_dataset.values[:, 0:13]
        self.y_train = train_dataset.values[:, 13]
        self.X_validation = validator_dataset.values[:, 0:13]
        self.y_validation = validator_dataset.values[:, 13]


def sklearn_lr(x, y, xv, yv):
    lr = LinearRegression()
    lr.fit(x, y)
    prediction = lr.predict(xv)
    print(mse(yv, prediction))


def BGD(X, y, Xv, yv):
    m, n = X.shape
    x0 = ones([m, 1])
    X = hstack((x0, X))
    m, n = X.shape
    theta_array = random.rand(14)
    alpha = 1e-4
    error = 1
    while error > 1e-12:
        h = dot(X, theta_array.reshape((n, 1)))
        sum_delta = dot(y - h.reshape((1, m)), X)
        theta_array = theta_array + alpha * sum_delta.flatten()
        error = norm(alpha * sum_delta.flatten())
        print(error)
    print(theta_array)

    m, n = Xv.shape
    Xv0 = ones([m, 1])
    Xv = hstack((Xv0, Xv))
    l2 = 0.0
    for i in range(m):
        l2 = l2 + square(dot(theta_array, Xv[i]) - yv[i])
    print(l2 / m)


def SGD(X, y, Xv, yv):
    m, n = X.shape
    X0 = ones([m, 1])
    X = hstack((X0, X))
    m, n = X.shape
    theta_array = random.rand(14)
    alpha = 0.01
    error = 1
    while error > 1e-6:
        theta_old = theta_array.copy()
        for i in range(m):
            h = dot(theta_array, X[i])
            theta_array = theta_array + alpha * (y[i] - h) * X[i]
            # for j in range(len(theta_array)):
            #     h = dot(theta_array, x[i])
            #     theta_array[j] = theta_array[j] + alpha * (y[i] - h) * x[i, j]
        error = norm(theta_old - theta_array)
        print(error)
    print(theta_array)

    m, n = Xv.shape
    xv0 = ones([m, 1])
    Xv = hstack((xv0, Xv))
    l2 = 0.0
    for i in range(m):
        l2 = l2 + square(dot(theta_array, Xv[i]) - yv[i])
    print(l2 / m)


def NE(x, y, xv, yv):
    theta_array = dot(inv(dot(x.T, x)), dot(x.T, y))
    print(theta_array)
    error = 0.0
    for i in range(len(yv)):
        error = error + square(dot(theta_array, xv[i]) - yv[i])
    print(error / len(yv))


def normalize(X):
    m, n = X.shape
    for j in range(n):
        min = X[:, j].min(axis=0)
        max = X[:, j].max(axis=0)
        if max - min != 0:
            X[:, j] = (X[:, j] - min) / (max - min)
        else:
            X[:, j] = 0
    return X


def standardize(X):
    m, n = X.shape
    for j in range(n):
        avg = X[:, j].mean(axis=0)
        std = X[:, j].std(axis=0)
        if std != 0:
            X[:, j] = (X[:, j] - avg) / std
        else:
            X[:, j] = 0
    return X


def main():
    data = Data()
    '''调用sklearn进行线性回归'''
    # 22.638256296587667
    sklearn_lr(data.X_train, data.y_train, data.X_validation, data.y_validation)
    '''批量梯度下降'''
    # 25.470440929155163
    # BGD(standardize(data.X_train), data.y_train, standardize(data.X_validation), data.y_validation)
    '''随机梯度下降'''
    # 22.298885141366405
    # SGD(standardize(data.X_train), data.y_train, standardize(data.X_validation), data.y_validation)
    '''最小二乘法'''
    # 24.292238175660408
    # NE(data.X_train, data.y_train, data.X_validation, data.y_validation)


if __name__ == '__main__':
    main()
