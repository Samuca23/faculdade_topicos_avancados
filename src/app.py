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