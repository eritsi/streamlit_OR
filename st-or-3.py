import pandas as pd
from pulp import *
from ortoolpy import addvars
import streamlit as st
import ast

st.header("生産計画問題")
st.markdown("""
資源量を守った上で最も利益を出すにはどのような生産計画を立案すべきか。  
""")

initial_sequence = "['石炭[kg/kg]', '電力[kWh/kg]', '労力[人時/kg]', '利益[万円/kg]'],\n [9, 4, 3, 7],\n [4, 5, 10,12]"
entered_sequence = st.text_area("Header & List of Products(最大化したい項目を最後に置く)", initial_sequence, height=250)
sequence = entered_sequence.splitlines()
col = list(ast.literal_eval(sequence[0]))
sequence = sequence[1:]
sequence = ''.join(sequence).strip()
sequence = list(ast.literal_eval(sequence))

table = pd.DataFrame(sequence, columns=col[0])
st.write(table.T)

initial_limit = "360, 200, 280"
entered_limit = st.text_input("制約条件： ".format(initial_limit), initial_limit)
limit_sequence = entered_limit.split(sep=',')
st.write(pd.DataFrame(limit_sequence, columns=['制約'], index=col[0][:-1]))

model = LpProblem(sense=LpMaximize) # 最適化モデル
table['変数'] = addvars(len(table)) # 生産量を表す変数を表に追加

model += lpDot(table.利益,table.変数) # 総利益を表す目的関数
for リソース in table.columns[:-2]: # 各製品ごとの処理
    model += lpDot(table[リソース],table.変数) <= df2[リソース] # 原料の使用量が在庫以下
    
model.solve() # ソルバで解を求める
table['結果'] = table.変数.apply(value) # 変数の値(結果)を表に追加
print('目的関数',value(model.objective)) # 目的関数の表示
table # 変数表の表示
# +
# weight = pd.DataFrame(sequence).T.tail(1).values.tolist()[0]
# value = pd.DataFrame(sequence).T.head(1).values.tolist()[0]
# capacity = int(entered_limit)

# maximized_value, selected_product = knapsack(weight, value, capacity)

# w=0
# for i in selected_product :
#     w +=weight[i]

# st.write("選ばれた品目: {}、　価値：{}€、 重量 : {} kg".format(selected_product, maximized_value, w))
