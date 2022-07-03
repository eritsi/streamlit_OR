import numpy as np
import matplotlib.pyplot as plt
# https://networkx.org/documentation/stable/index.html
import networkx as nx
import streamlit as st
import ast
from PIL import Image

st.header("最短経路問題")
st.markdown("""
あるネットワークにおいて、各点と隣接する点同士の移動距離が分かっているとします。
この時ある始点から終点までの距離が最短になるルートを求めたい場合、どのように移動経路を設定すれば良いでしょうか？
""")

# +
initial_sequence = "(0, 1, 3), (0, 2, 1), (1, 2, 3), (1, 3, 2), (2, 3, 6), (2, 4, 5), (2, 3, 6), (3, 4, 2), (1, 5, 9), (3, 5, 4)"
entered_sequence = st.text_input("Enter Network Sequence (from, to, time)", initial_sequence)
sequence_list = list(ast.literal_eval(entered_sequence))

fig = plt.figure(figsize=(12,5))
G = nx.Graph()
G.add_weighted_edges_from(
    sequence_list, 
    weight='weight')
#nx.add_path(g, nx.dijkstra_path(g, 0, 5), color="red")


#　始点終点のラベル変更
# 2つ目のパラメータが最大の集合を取り出し、その2つ目のパラメータを返す
terminal = max(sequence_list, key=(lambda x: x[1]))[1]
G = nx.relabel_nodes(G, {0: "S", terminal: "T"})

# 距離表示
pos = nx.spring_layout(G, k=1, pos={"S":[0, 0], "T":[1, 0]}, fixed=["S", "T"])
edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
options = {
    "font_size": 15,
    "node_size": 250,
    "node_color": "c",
    "edgecolors": "black",
    "linewidths": 1,
    "width": 1,
}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, verticalalignment='baseline', font_size=15)
nx.draw_networkx(G, pos, **options)


# 表示
ax = plt.gca()
ax.margins(0.20)
plt.axis("off")
plt.show()
st.pyplot(fig)
# -

if st.button('最短経路を表示'):
    st.write(nx.dijkstra_path(G, 'S', 'T'))
    st.write("経路の合計時間:　{}".format(nx.dijkstra_path_length(G, 'S', 'T')))

st.header("全経路を表示")
st.markdown("""
全経路を表示して最短経路が正しかったかを確かめてみましょう。また、最長経路も探してみましょう。""")

if st.button('全経路を表示'):
#     st.write(nx.single_source_bellman_ford(G,'S', 'T'))
    # 全経路と長さ
    for path in nx.all_simple_paths(G, 'S', 'T'):
        st.write('path: {} length:{}'.format(path, nx.path_weight(G, path, weight="weight")))

st.header("解答")
st.image(Image.open('img/or-network.png'))

st.header("巡回セールスマン問題")
st.markdown("""
始点と終点を定めず、最短となる一筆書き経路を求めてみましょう。""")

if st.button('最短一筆書き経路を表示'):
#     st.write(nx.single_source_bellman_ford(G,'S', 'T'))
    # 始点・終点を定めない巡回セールスマン問題
    tsp = nx.approximation.traveling_salesman_problem
    st.write(tsp(G, weight='weight', cycle=False))