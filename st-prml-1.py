# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from prml.preprocess import PolynomialFeature
from prml.linear import (
    LinearRegression,
    RidgeRegression,
    BayesianRegression
)
import streamlit as st

# +
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('1.Polynomial Curve Fitting')
SEED = st.sidebar.slider('Random Seed', 0, 1500, 1234)

selected_sample_size = st.sidebar.slider('サンプル点数', 0, 50, 10)

selected_std = st.sidebar.slider('標準偏差//sin(2πx)からのばらつき', 0.0, 1.0, 0.25)

# -

np.random.seed(SEED)


# ### 線形回帰
# 次数を変えたり、サンプルを変えながら線形回帰を理解する

# +
def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return np.sin(2 * np.pi * x)

x_train, y_train = create_toy_data(func, selected_sample_size, selected_std)
x_test = np.linspace(0, 1, 100)
y_test = func(x_test)

f, ax = plt.subplots(figsize=(7, 5))
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.legend()
plt.show()
st.header("1.Polynomial Curve Fitting")
st.markdown("""
    sin(2πx)を正解とし、その周りにランダムサンプルを生成する。標準偏差をサイドバーから指定可能。
    """)
st.pyplot(f)

# +
# Sidebar
# Header of Specify Input Parameters

st.sidebar.header('2.次数')
selected_degree = st.sidebar.slider('Set how many polinominal dimensions', 0, 10, 3)

# +
f2 = plt.figure(figsize=(12,5))
for i, degree in enumerate([0, 1, selected_degree, 9]):
    plt.subplot(int("22{}".format(i + 1)))
    feature = PolynomialFeature(degree)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y = model.predict(X_test)

    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="fitting")
    plt.ylim(-1.5, 1.5)
    plt.annotate("M={}".format(degree), xy=(0.05, 1))
plt.legend(bbox_to_anchor=(1.05, 0.64), loc=2, borderaxespad=0.)
st.header("2.次数を変えた線形回帰")
st.markdown("""
    左下のグラフについて、サイドバーの次数設定によって特徴量関数として何次元の多項式まで使用するかを選択可能。  
    次数を上げる{1, x, x^2, x^3, x^4,...}につれ、表現力が上昇しsin(2πx)に近づく。 
    しかし次数を上げすぎるとサンプル点を通ることを重視しすぎ、それ以外の部分でsin(2πx)から乖離することが見て取れる。  
    この状態を過学習という。
    """)

plt.show()
st.pyplot(f2)


# -

# ### 過学習
# 次数が上がるにつれ精度が改善するものの、上げすぎると過学習になっていることを示す

# +
def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

training_errors = []
test_errors = []

for i in range(10):
    feature = PolynomialFeature(i)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y = model.predict(X_test)
    training_errors.append(rmse(model.predict(X_train), y_train))
    test_errors.append(rmse(model.predict(X_test), y_test + np.random.normal(scale=0.25, size=len(y_test))))

f3, ax3 = plt.subplots(figsize=(7, 5))
plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()
st.header("3.次数ごとの線形回帰精度")
st.markdown("""
    次数を上げるほど、training dataとの誤差は少なくなっていくが、test data(正解、目指している関数)との 
    誤差は次数を上げすぎると悪化していることがわかる。汎化性能が阻害されている状態。
    """)
st.pyplot(f3)
# -

# ### 正則化
# 正則化により、高い次数において過学習を回避できることを示す

# +
# Sidebar
# Header of Specify Input Parameters

st.sidebar.header('4.正則化')
selected_Ridge = st.sidebar.slider('リッジのα [1e-3 *]', 0, 10, 1)
selected_Ridge= 1e-3 * selected_Ridge

# +
feature = PolynomialFeature(selected_degree)
X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = RidgeRegression(alpha=selected_Ridge)
model.fit(X_train, y_train)
y = model.predict(X_test)

f4, ax4 = plt.subplots(figsize=(7, 5))
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="fitting")
plt.ylim(-1.5, 1.5)
plt.legend()
plt.annotate("M={}".format(selected_degree), xy=(0.05, 1))
plt.show()
st.header("4.正則化項を導入した線形回帰")
st.markdown("""
    先ほど過学習が起きていた次数(特徴量関数の数)においても、  
    過学習に対する罰則となるL2正則化項を導入（＝リッジの線形回帰）すると過学習が抑制される。
    """)
st.pyplot(f4)
# -

# ### ベイジアン回帰
# ベイズ的考え方を導入

# +
model = BayesianRegression(alpha=2e-3, beta=2)
model.fit(X_train, y_train)

y, y_err = model.predict(X_test, return_std=True)
f5, ax5 = plt.subplots(figsize=(7, 5))
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, y, c="r", label="mean")
plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
plt.xlim(-0.1, 1.1)
plt.ylim(-1.5, 1.5)
plt.annotate("M={}".format(selected_degree), xy=(0.8, 1))
plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
plt.show()
st.header("5.ベイズ導入した線形回帰")
st.markdown("""
    ベイズにおいては、confidence levelを表現するのが得意。
    """)
st.pyplot(f5)
