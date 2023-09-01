import pandas as pd
import streamlit as st
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster

st.set_page_config(layout="wide")
st.title('App - Tópicos Avançados')

## Leitura dos banco de dados em cache
@st.cache_data
def load_database():
    return pd.read_feather('../dados/gs.feather'), \
        pd.read_feather('../dados/localizacao.feather')

gs, coordenadas = load_database()

## Criação das opções com base em tabs
taberp, tabbi, tabstore = st.tabs(['Sistema Interno', 'Gestão', 'E-Commerce'])

with taberp:
    st.header('Dados do Sistema Interno')
    consumidor = st.selectbox(
        'Selecione o consumidor',
        gs['Customer ID'].unique()
    )
    gs_con = gs[gs['Customer ID'] == consumidor]
    data = gs_con.merge(
        coordenadas.drop_duplicates(),
        left_on=['City', 'Country'],
        right_on=['cidade', 'pais'],
        how='left'
    )
    data = data.fillna(0)
    with st.expander('Pedidos:'):
        st.table(data[
                ['Order Date','Product Name','Quantity','Sales','Profit','Country','lng','lat']
            ]
        )
        on = st.checkbox('Mostrar no mapa')
        ## if st.checkbox('Mostrar Mapas de Localização dos Pedidos'):
        if on:
            m = folium.Map(location=[0, 0], tiles='openstreetmap', zoom_start=2)
            for id,row in data.iterrows():
                folium.Marker(location=[row['lat'],row['lng']], popup=row['Profit']).add_to(m)
            folium_static(m)


            m2 = folium.Map(location=[0,0], tiles='cartodbpositron', zoom_start=2)
            mc = MarkerCluster()
            for idx, row in data.iterrows():
                mc.add_child(folium.Marker([row['lat'], row['lng']],popup=row['Country']))
            m2.add_child(mc)
            folium_static(m2)