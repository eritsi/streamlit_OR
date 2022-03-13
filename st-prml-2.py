# https://www.slideshare.net/TakutoKimura/prml21-222425
# slide 3,18
import numpy as np
import matplotlib.pyplot as plt
import ast

from prml.rv import (
    Bernoulli,
    Beta,
    Categorical,
    Dirichlet,
    Gamma,
    Gaussian,
    MultivariateGaussian,
    MultivariateGaussianMixture,
    StudentsT,
    Uniform
)

import streamlit as st


st.header('1.ベルヌーイ分布')

# -


# ### コイン投げ
# ベルヌーイ分布、二項分布とその共役分布であるベータ分布まで

# +

st.markdown("""
1. コイン投げはベルヌーイ分布に従う
1. 数回のコイン投げの結果から、コインの'歪み'に当たるパラメータμを同定
1. パラメータμは一つに定まるという考え方:最尤推定、Maximum Likelyhood
""")
initial_sequence = "[0, 1, 1, 1]"
entered_sequence = st.text_input("コイン投げの結果を入力 0:表、1:裏", initial_sequence)
sequence = ast.literal_eval(entered_sequence)

model = Bernoulli()
model.fit(np.array(sequence))
st.write(model)

st.header('2.ベータ分布')
st.markdown("""
ベルヌーイ分布の共役事前分布はベータ分布であり、そのshape factor [a,b]を設定を設定しましょう。  
試行結果から、パラメータμの確率分布を図示します。  

a, b がそれぞれ2以上の時、aは0,bは1の観測回数と等しく見做すことができます。  
また、事前分布と事後分布は同じ形である（=共役性)ため、逐次更新ができます。
""")
f = plt.figure(figsize=(12,5))
x = np.linspace(0, 1, 100)
initial_sequence_beta = "[2, 3]"
entered_sequence_beta = st.text_input("ベータ分布のパラメータ[a,b]", initial_sequence_beta)
sequence_beta = ast.literal_eval(entered_sequence_beta)

for i, [a, b] in enumerate([[0.1, 0.1], [1, 1], [sequence_beta[0], sequence_beta[1]], [8, 4]]):
    plt.subplot(int("22{}".format(i + 1)))
    beta = Beta(a, b)
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    plt.xlabel("μ")
    plt.ylabel("probability density")
    plt.plot(x, beta.pdf(x))
    plt.annotate("a={}".format(a), (0.1, 2.5))
    plt.annotate("b={}".format(b), (0.1, 2.1))
plt.show()
st.pyplot(f)

st.header('3.事前分布と事後分布')
st.markdown("""
事前分布が、数回のコイン投げ結果により更新されていく様が見れます。  
事前分布と結果に大きな乖離がある場合(例 [a,b]=[8,4])でどのように結果が更新されるか観察しましょう。
""")
f2 = plt.figure(figsize=(12,5))
beta = Beta(sequence_beta[0], sequence_beta[1])
plt.subplot(2, 1, 1)
plt.xlim(0, 1)
plt.ylim(0, np.ceil(max(beta.pdf(x))))
plt.ylabel("probability density")
plt.plot(x, beta.pdf(x))
plt.annotate("prior", (0.1, 1.5))

model = Bernoulli(mu=beta)
model.fit(np.array(sequence))
plt.subplot(2, 1, 2)
plt.xlim(0, 1)
plt.ylim(0, np.ceil(max(model.mu.pdf(x))))
plt.xlabel("μ")
plt.ylabel("probability density")
plt.plot(x, model.mu.pdf(x))
plt.annotate("posterior", (0.1, 1.5))

plt.show()
st.pyplot(f2)

st.header('4.最尤推定とベイズ推定')
st.markdown("""
1が出る確率を予測推定します。  
最尤推定では、ごく少ない試行で極端な結果が出た場合に、推定結果も極端(=過学習)になります。  
一方でベイズ推論においては「結果に飛びつく」ことなく、徐々に事後分布を矯正
しながら推定が進みます。

例えばコイン投げのたった１回の試行結果が裏(最初のインプットを[1]に)であった時、
各々の推定結果を用いて10000回の試行結果を推論するとどのような結果になるかを
見てみましょう。
""")
SEED = st.slider('Random Seed', 0, 1500, 1234)
np.random.seed(SEED)

st.write("Maximum likelihood estimation")
model = Bernoulli()
model.fit(np.array(sequence))
st.write("{} out of 10000 is 1".format(model.draw(10000).sum()))

st.write("Bayesian estimation")
model = Bernoulli(mu=Beta(sequence_beta[0], sequence_beta[1]))
model.fit(np.array(sequence))
st.write("{} out of 10000 is 1".format(model.draw(10000).sum()))

st.header('5.カテゴリカル分布')
st.markdown("""
カテゴリカル分布はベルヌーイ分布を多次元変数へ一般化したものです。  
""")

initial_sequence_dice = "[3, 2, 5, 6]"
entered_sequence_dice = st.text_input("サイコロの結果を入力", initial_sequence_dice)
sequence_dice = ast.literal_eval(entered_sequence_dice)

# xを6クラスの1-of-K型配列に変換
def incr(lst, i):
    return [x+i for x in lst]

x_class = np.array(incr(sequence_dice, -1))
x_class = np.identity(6, dtype = "int32")[x_class]

model = Categorical()
model.fit(np.array(x_class))
st.write("カテゴリカル分布のμは、各変数の最尤推定結果を示しています")
st.write(model)

st.header('6.ディリクレ分布')
st.markdown("""
カテゴリカル分布の共役分布になります。  
事前分布のパラメータαを[1,1,1,1,1,1]とおきましょう。  
αは各変数の観測回数に合致します。
""")

mu = Dirichlet(alpha=np.ones(6))
model = Categorical(mu=mu)
st.write(model)

model.fit(np.array(x_class))
st.write(model)

st.header('7.スチューデントのt分布')
st.markdown("""
正規分布の限界  
正規分布に従うサンプルを20用意し、外れ値を3つ含めます。  
""")
f3 = plt.figure(figsize=(12,5))
X = np.random.normal(size=20)
X = np.concatenate([np.random.normal(loc=20., size=3), X])
st.write(X)
plt.hist(X.ravel(), bins=50, density=1., label="samples")

students_t = StudentsT()
gaussian = Gaussian()

gaussian.fit(X)
students_t.fit(X)

st.write("""
正規分布を当てはめた場合
""")
st.write(gaussian)
st.write("""
Student's t分布を当てはめた場合
""")
st.write(students_t)

x = np.linspace(-5, 25, 1000)
plt.plot(x, students_t.pdf(x), label="student's t", linewidth=2)
plt.plot(x, gaussian.pdf(x), label="gaussian", linewidth=2)
plt.legend()
plt.show()
st.pyplot(f3)
