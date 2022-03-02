import numpy as np
import pandas as pd
import streamlit as st
import ast
from ortoolpy import knapsack

st.header("ナップサック問題")
st.markdown("""
最も価値のある組み合わせをナップサックに入れましょう。  
""")

initial_sequence = "[5, 17],\n[50, 56],\n[50, 59],\n[64, 80],\n[46, 64],\n[50, 75]"
entered_sequence = st.text_area("List of Products: [value, weight]", initial_sequence, height=250)
sequence = entered_sequence.splitlines()
sequence = ''.join(sequence)
sequence = list(ast.literal_eval(sequence))

st.write(pd.DataFrame(sequence, columns=["value[€]","weight[kg]"]))

# +
initial_limit = "190"
entered_limit = st.text_input("最大重量 : ".format(initial_limit), initial_limit)

weight = pd.DataFrame(sequence).T.tail(1).values.tolist()[0]
value = pd.DataFrame(sequence).T.head(1).values.tolist()[0]
capacity = int(entered_limit)

maximized_value, selected_product = knapsack(weight, value, capacity)

w=0
for i in selected_product :
    w +=weight[i]

st.write("選ばれた品目 : {}、 価値 : {}€、 重量 : {} kg".format(selected_product, maximized_value, w))

