# https://www.slideshare.net/sotetsukoyamada/prml3332
# slide 3,5,7
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from prml.preprocess import GaussianFeature, PolynomialFeature, SigmoidalFeature
from prml.linear import (
    BayesianRegression,
    EmpiricalBayesRegression,
    LinearRegression,
    RidgeRegression
)
import streamlit as st

# +
# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('1.Curve Fitting')
SEED = st.sidebar.slider('Random Seed', 0, 1500, 1234)
np.random.seed(SEED)

selected_sample_size = st.sidebar.slider('サンプル点数', 5, 50, 10)
selected_Basis_Dim = st.sidebar.slider('基底関数の次元', 1, 20, 8)

def create_toy_data(func, sample_size, std, domain=[0, 1]):
    x = np.linspace(domain[0], domain[1], sample_size)
    np.random.shuffle(x)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t



st.header('1. Linear Basis Function Models')
st.write("""基底関数の可視化  
左から多項式、ガウス基底、シグモイド基底
""")
# -


# ### 線型基底関数
# 1章では多項式の基底関数だけだったが、これを拡張する

# +

x = np.linspace(-1, 1, 100)
X_polynomial = PolynomialFeature(11).transform(x[:, None])
X_gaussian = GaussianFeature(np.linspace(-1, 1, 11), 0.1).transform(x)
X_sigmoidal = SigmoidalFeature(np.linspace(-1, 1, 11), 10).transform(x)

f1 = plt.figure(figsize=(20, 5))
for i, X in enumerate([X_polynomial, X_gaussian, X_sigmoidal]):
    plt.subplot(int("13{}".format(i + 1)))
    for j in range(12):
        plt.plot(x, X[:, j])
st.pyplot(f1)

st.subheader("1.1 Maximum likelihood and least squares")

def sinusoidal(x):
    return np.sin(2 * np.pi * x)

x_train, y_train = create_toy_data(sinusoidal, selected_sample_size, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = sinusoidal(x_test)

# Pick one of the three features below
selected_Basis_Func = st.sidebar.selectbox('which Basis Function Model to use?',
                                            ['多項式','ガウス','シグモイド'], 
                                            index=1)
if selected_Basis_Func == '多項式':
    feature = PolynomialFeature(selected_Basis_Dim)
elif selected_Basis_Func == 'ガウス':
    feature = GaussianFeature(np.linspace(0, 1, selected_Basis_Dim), 0.1)
elif selected_Basis_Func == 'シグモイド':
    feature = SigmoidalFeature(np.linspace(0, 1, selected_Basis_Dim), 10)

X_train = feature.transform(x_train)
X_test = feature.transform(x_test)
model = LinearRegression()
model.fit(X_train, y_train)
y, y_std = model.predict(X_test, return_std=True)

f2 = plt.figure(figsize=(12, 5))
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, label="$\sin(2\pi x)$")
plt.plot(x_test, y, label="prediction")
plt.fill_between(
    x_test, y - y_std, y + y_std,
    color="orange", alpha=0.5, label="std.")
plt.legend()
plt.show()
st.pyplot(f2)

st.header('1.4 Regularized least squares')
st.write("""正則化二乗誤差関数  
リッジL2項の導入による過学習の抑制
""")


model = RidgeRegression(alpha=1e-3)
model.fit(X_train, y_train)
y = model.predict(X_test)

f3 = plt.figure(figsize=(12,5))
plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.plot(x_test, y_test, label="$\sin(2\pi x)$")
plt.plot(x_test, y, label="prediction")
plt.legend()
plt.show()
st.pyplot(f3)

st.header('2. The Bias-Variance Decomposition')
st.write("""バイアス - バリアンス分解  
ベイズ回帰におけるパラメータαを変えながら、左側には100回の予測の試行(うち20回分をプロット)、右側には100回の予測結果の平均を表示している。  
下に行くほどvarianceは高いがbiasは低いことがわかる。つまりこの二つはトレードオフの関係。
""")
# -
# https://www.slideshare.net/sotetsukoyamada/prml3332
# slide 20, 26

f4 = plt.figure(figsize=(20, 15))
for j, a in enumerate([1e2, 1., 1e-9]):
    y_list = []
    plt.subplot(int("32{}".format(j * 2 + 1)))
    for i in range(100):
        x_train, y_train = create_toy_data(sinusoidal, 25, 0.25)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)
        model = BayesianRegression(alpha=a, beta=1.)
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        y_list.append(y)
        if i < 20:
            plt.plot(x_test, y, c="orange")
    plt.ylim(-1.5, 1.5)
    plt.annotate("alpha={}".format(a), xy=(0.05, -1))

    plt.subplot(int("32{}".format(j * 2 + 2)))
    plt.plot(x_test, y_test)
    plt.plot(x_test, np.asarray(y_list).mean(axis=0))
    plt.ylim(-1.5, 1.5)
    plt.show()
st.pyplot(f4)

st.header('3. Bayesian Linear Regression')
# -

st.subheader('3.1. Parameter distribution')
# https://www.slideshare.net/taki0313/prml-sec3
# slide 32
st.write("""  
y = w0 + w1 * x  
の周りにランダムに発生させたサンプル点を徐々にモデルへ与える。サンプル点の具体的な数値は以下の表の通り。  
""")

selected_w0 = st.sidebar.slider('Y切片(w0)', -1.0, 1.0, -0.3)
selected_w1 = st.sidebar.slider('傾き(w1)', -1.0, 1.0, 0.5)

def linear(x):
    return selected_w0 + selected_w1 * x

x_train, y_train = create_toy_data(linear, selected_sample_size, 0.1, [-1, 1])
x = np.linspace(-1, 1, 100)
w0, w1 = np.meshgrid(
    np.linspace(-1, 1, 100),
    np.linspace(-1, 1, 100))
w = np.array([w0, w1]).transpose(1, 2, 0)

st.dataframe([x_train, y_train])

st.write("""  
説明の関係上、モデルの基底関数は多項式1次に固定。(恣意的)   
左の図ではw0,w1の確率分布が絞られていく様子を、右の図では線型回帰結果が真の直線に近づいていく様子を示す。
""")

feature = PolynomialFeature(degree=1)
X_train = feature.transform(x_train)
X = feature.transform(x)
model = BayesianRegression(alpha=1., beta=100.)


f5 = plt.figure(figsize=(10, 25))
for i, [begin, end] in enumerate([[0, 0], [0, 1], [1, 2], [2, 3], [3, selected_sample_size]]):
    model.fit(X_train[begin: end], y_train[begin: end])
    plt.subplot(5, 2, i * 2 + 1)
    plt.scatter(selected_w0, selected_w1, s=200, marker="x")
    plt.contour(w0, w1, multivariate_normal.pdf(w, mean=model.w_mean, cov=model.w_cov))
    plt.gca().set_aspect('equal')
    plt.xlabel("$w_0$")
    plt.ylabel("$w_1$")
    plt.title("prior/posterior")

    plt.subplot(5, 2, i * 2 + 2)
    plt.scatter(x_train[:end], y_train[:end], s=100, facecolor="none", edgecolor="steelblue", lw=1)
    plt.plot(x, model.predict(X, sample_size=6), c="orange")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

x_test = np.linspace(-1, 1, 10)
y_test = linear(x_test)

# 最後のプロットだけ真の値を追加
plt.plot(x_test, y_test, c="g", label="{} + {} * x".format(selected_w0,selected_w1))
plt.legend(bbox_to_anchor=(0.5, 0.95), loc=2, borderaxespad=0.)
st.pyplot(f5)

st.subheader('3.2. Predictive distribution')
# https://www.slideshare.net/taki0313/prml-sec3
# slide 36
st.write("""   
""")

# x_train, y_train = create_toy_data(sinusoidal, 25, 0.25)
x_train, y_train = create_toy_data(sinusoidal, selected_sample_size, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = sinusoidal(x_test)

if selected_Basis_Func == '多項式':
    feature = PolynomialFeature(selected_Basis_Dim)
elif selected_Basis_Func == 'ガウス':
    feature = GaussianFeature(np.linspace(0, 1, selected_Basis_Dim), 0.1)
elif selected_Basis_Func == 'シグモイド':
    feature = SigmoidalFeature(np.linspace(0, 1, selected_Basis_Dim), 10)
# feature = GaussianFeature(np.linspace(0, 1, 9), 0.1)

X_train = feature.transform(x_train)
X_test = feature.transform(x_test)

model = BayesianRegression(alpha=1e-3, beta=2.)
x = np.linspace(-1, 1, 100)
X = feature.transform(x)
f6 = plt.figure(figsize=(10, 20))
for i, [begin, end] in enumerate([[0, 1], [1, 2], [2, 4], [4, 8], [8, selected_sample_size]]):
    model.fit(X_train[begin: end], y_train[begin: end])
    plt.subplot(5, 2, i * 2 + 1)
    y, y_std = model.predict(X_test, return_std=True)
    plt.scatter(x_train[:end], y_train[:end], s=100, facecolor="none", edgecolor="steelblue", lw=2)
    plt.plot(x_test, y_test, label="y_true: sinusoid")
    plt.plot(x_test, y, label="y_pred")
    plt.fill_between(x_test, y - y_std, y + y_std, color="orange", alpha=0.5)
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.legend()

    plt.subplot(5, 2, i * 2 + 2)
    plt.scatter(x_train[:end], y_train[:end], s=100, facecolor="none", edgecolor="steelblue", lw=1)
    plt.plot(x, model.predict(X, sample_size=6), c="orange")
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.show()
st.pyplot(f6)


st.subheader('3.5. The Evidence Approximation')
# https://www.slideshare.net/taki0313/prml-sec3
# slide 54
st.write("""説明の関係上、モデル基底関数は多項式に固定。   
""")

def cubic(x):
    return x * (x - 5) * (x + 5)

x_train, y_train = create_toy_data(cubic, 30, 10, [-5, 5])
x_test = np.linspace(-5, 5, 100)
evidences = []
models = []

def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

training_errors = []
test_errors = []
for i in range(selected_Basis_Dim + 1):
    feature = PolynomialFeature(degree=i)
    X_train = feature.transform(x_train)
    model = EmpiricalBayesRegression(alpha=100., beta=100.)
    model.fit(X_train, y_train, max_iter=100)
    evidences.append(model.log_evidence(X_train, y_train))
    models.append(model)
    training_errors.append(rmse(model.predict(X_train), y_train))
    X_test = feature.transform(x_test)
    test_errors.append(rmse(model.predict(X_test), cubic(x_test)))

# EvidenceがMaxとなる次数のモデルを採用している
degree = np.nanargmax(evidences)
regression = models[degree]

X_test = PolynomialFeature(degree=int(degree)).transform(x_test)
y, y_std = regression.predict(X_test, return_std=True)



f7 = plt.figure(figsize=(20,15))
plt.subplot(3, 1, 3)
plt.scatter(x_train, y_train, s=50, facecolor="none", edgecolor="steelblue", label="observation")
plt.plot(x_test, cubic(x_test), label="x(x-5)(x+5)")
plt.plot(x_test, y, label="prediction")
plt.title("Best chosed Model predicition // chosen degree={}".format(degree))
plt.fill_between(x_test, y - y_std, y + y_std, alpha=0.5, label="std", color="orange")
plt.legend()

plt.subplot(3, 1, 1)
plt.plot(evidences)
plt.title("Model evidence (Max is the best model)")
plt.xlabel("degree")
plt.ylabel("log evidence")

plt.subplot(3, 1, 2)
plt.plot(training_errors, 'o-', mfc="none", mec="b", ms=10, c="b", label="Training")
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.title("Degree of model versus RMSE")
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()
st.pyplot(f7)
