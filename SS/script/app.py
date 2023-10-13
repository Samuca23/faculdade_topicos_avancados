## pacotes de tratamento de dados, interface, gráfico e mapas
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.express as px
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from streamlit_extras.metric_cards import style_metric_cards

st.set_page_config(layout="wide")
st.title('App - Tópicos Avançados')

style_metric_cards(
    border_left_color="#3D5077",
    background_color="#F0F2F6",
    border_size_px=3,
    border_color = "#CECED0",
    border_radius_px = 10,
    box_shadow=True
)

## Leitura dos banco de dados em cache
@st.cache_data
def load_database():
    return pd.read_feather('../dados/ss.feather'), \
        pd.read_feather('../dados/knn_estado.feather'), \
        pd.read_feather('../dados/probabilidade_estado.feather'), \
        pd.read_feather('../dados/classificacao_consumidor.feather'), \
        pd.read_feather('../dados/clusterizacao_estado.feather')


gs, knn_estado, prb_pai, cla_con, clu_pai = load_database()

## Criação das opções com base em tabs
taberp, tabbi, tabstore = st.tabs(['Sistema Interno', 'Gestão', 'E-Commerce'])

with taberp:
    st.header('Dados do Sistema Interno')
    consumidor = st.selectbox(
        'Selecione o consumidor',
        gs['Customer ID'].unique()
    )
    gs_con = gs[gs['Customer ID'] == consumidor]
    cla_con_con = cla_con[cla_con['Customer ID'] == consumidor].reset_index()
    st.dataframe(gs_con[['Customer Name', 'Segment']].drop_duplicates())
    cl1, cl2, cl3, cl4 = st.columns(4)
    cl1.metric('Score', round(cla_con_con['score'][0],4))
    cl2.metric('Classe', round(cla_con_con['classe'][0],4))
    cl3.metric('Rank', round(cla_con_con['rank'][0],4))
    cl4.metric('Lucro', round(cla_con_con['lucro'][0],4))
    cl1.metric('Valor Total Comprado', round(gs_con['Sales'].sum(),2))
    cl2.metric('Valor Lucro', round(gs_con['Profit'].sum(),2))
    cl3.metric('Valor Médio Comprado', round(gs_con['Sales'].mean(),2))
    cl4.metric('Quantidade Comprada', round(gs_con['Quantity'].sum(),2))
    st.write(gs_con['State'].values[0])
    st.dataframe(
        prb_pai[prb_pai['State'] == gs_con['State'].values[0]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "prob_lucro": st.column_config.ProgressColumn("Prob Lucro", format="%.2f", min_value=0, max_value=1),
            "prob_prejuizo": st.column_config.ProgressColumn("Prob Prejuízo", format="%.2f", min_value=0, max_value=1),
        }
    )
    prob = st.toggle('Similares')
    if prob:
        st.dataframe(
            knn_estado[knn_estado['referencia'] == gs_con['State'].values[0]].merge(
                prb_pai, left_on='vizinho', right_on='State', how='left')[
                ['State','Sales','Quantity','Profit','prob_prejuizo','prob_lucro']
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "prob_lucro": st.column_config.ProgressColumn("Prob Lucro", format="%.2f", min_value=0, max_value=1),
                "prob_prejuizo": st.column_config.ProgressColumn("Prob Prejuízo", format="%.2f", min_value=0, max_value=1),
            }
        )
    clus = st.toggle('Clusters')
    if clus:
        clu_pai_cli = clu_pai[clu_pai['referencia'] == gs_con['State'].values[0]]
        st.write('Dados do Cluster do Estado')
        st.dataframe(clu_pai_cli[
            ['cluster', 'clm_lucro', 'clm_vendas', 'clm_qtde', \
                'clf_vendas', 'cls_lucro', 'clr_dias']],
            hide_index=True,
            use_container_width=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c2.metric('Montante de Lucro',
                  clu_pai_cli['m_lucro'].values[0],
                  delta=clu_pai_cli['m_lucro'].values[0] - clu_pai_cli['clm_lucro'].values[0])
        c3.metric('Montante de Vendas',
                  clu_pai_cli['m_vendas'].values[0],
                  delta=clu_pai_cli['m_vendas'].values[0] - clu_pai_cli['clm_vendas'].values[0])
        c4.metric('Montante de Quantidade',
                  clu_pai_cli['m_qtde'].values[0],
                  delta=clu_pai_cli['m_qtde'].values[0] - clu_pai_cli['clm_qtde'].values[0])
        c1.metric('Periodicidade em Dias', clu_pai_cli['r_dias'].values[0],
                  delta=clu_pai_cli['r_dias'].values[0] - clu_pai_cli['clr_dias'].values[0],
                  delta_color='inverse')
        c2.metric('Frequencia de Vendas', clu_pai_cli['f_vendas'].values[0],
                  delta=clu_pai_cli['f_vendas'].values[0] - clu_pai_cli['clf_vendas'].values[0])
        c3.metric('Frequencia de Lucro', clu_pai_cli['f_lucro'].values[0],
                  delta=clu_pai_cli['f_lucro'].values[0] - clu_pai_cli['cls_lucro'].values[0])
    pedidos = st.toggle('Pedidos')
    if pedidos:
        st.dataframe(gs_con[
                [
                    'Order Date','Category','Sub-Category','Product Name',
                    'City', 'State', 'Region',
                    'Quantity','Sales','Profit'
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Sales": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                "Profit": st.column_config.NumberColumn("Profit($)", format="$%.2f"),
            }
        )
    cross = st.toggle('Cross Table')
    if cross:
        cln1, cln2, cln3 = st.columns(3)
        variaveis = [
            'Category','Sub-Category','Product Name',
            'City', 'State', 'Region'
        ]
        linha = cln1.multiselect('Linha: ', variaveis)
        coluna = cln2.multiselect('Coluna: ', variaveis)
        valor = cln3.selectbox('Valor: ', ['Quantity','Sales','Profit'])
        if len(linha) > 0 and len(coluna) > 0:
            st.dataframe(
                gs_con.pivot_table(
                    index=linha,
                    columns=coluna,
                    values=valor,
                    aggfunc='sum',
                    fill_value=0
                ).style.background_gradient(cmap='Greys'),
                use_container_width=True,
            )