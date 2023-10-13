## pacotes de tratamento de dados e modelo de inteligência
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pyod.models.knn import KNN
from sklearn.linear_model import LogisticRegression

## funções auxiliares
from funcoes import zscore, rfm_variables, fit_data, outliers_detection
## Leitura do banco de dados
data = pd.read_feather('../dados/ss.feather')
## Variáveis para utilização no modelo conforme RFM Analysis
variaveis = [
    'f_vendas', 'f_lucro', 'm_lucro', 'm_qtde',
    'm_vendas', 'r_dias'
]

print('Classificação do Consumidor...')
gr_con = data.groupby(
    [
        'Customer ID'
    ]
)[
    [
        'Sales',
        'Quantity',
        'Profit'
    ]
].mean().reset_index()
for col in ['Sales', 'Quantity', 'Profit']:
    gr_con = zscore(gr_con, 'Customer ID', col, 'z'+col)
gr_con['score'] = gr_con['zSales'] + \
                + gr_con['zQuantity'] \
                + gr_con['zProfit']
media_score = gr_con['score'].mean()
dpadr_score = gr_con['score'].std()
gr_con['classe'] = gr_con['score'].apply(lambda x : int((x - media_score) / dpadr_score) + 3)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 0 if x < 0 else x)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 6 if x > 6 else x)
gr_con['rank'] = gr_con['score'].rank(ascending=False)
gr_con['lucro'] = gr_con['Profit'].apply(lambda x : 0 if x < 0 else 1)
gr_con.to_feather('../dados/classificacao_consumidor.feather')

print('Cálculo da probabilidade de lucro por estado...')
gr_estado = data.groupby('State')[['Sales','Quantity','Profit']].mean().copy()
gr_estado['Lucro'] = gr_estado['Profit'].apply(lambda x : 0 if x < 0 else 1)
X_Train = gr_estado.drop(columns=['Lucro'], axis=1)
X_Test = gr_estado.drop(columns=['Lucro'], axis=1)
y_Train = gr_estado['Lucro']
y_Test = gr_estado['Lucro']
sc_x = StandardScaler()
X_Train = sc_x.fit_transform(X_Train)
X_Test = sc_x.fit_transform(X_Test)
logreg = LogisticRegression(solver="lbfgs", max_iter=500)
logreg.fit(X_Train, y_Train)
pred_logreg = logreg.predict(X_Test)
pred_proba = logreg.predict_proba(X_Test)
gr_estado['previsao'] = pred_logreg
lista_proba = pred_proba.tolist()
lista_proba = pd.DataFrame(
    lista_proba, columns = ['prob_prejuizo', 'prob_lucro']
)
gr_estado = gr_estado.reset_index()
gr_estado = pd.merge(gr_estado, lista_proba, left_index=True, right_index=True)
gr_estado.to_feather('../dados/probabilidade_estado.feather')

print('Cálculo da associação por estado')
original = fit_data(data, 'State')
original = original.fillna(0)
base = original[variaveis]
vizinhos = NearestNeighbors(n_neighbors=min(4, len(base))).fit(base)
similares = []
for index, row in original.iterrows():
    #print('Referencia: {0}'.format(row['referencia']))
    #print('Referencias Similares:')
    original_referencia = original[
        original['referencia'] == row['referencia']][variaveis]
    similar = vizinhos.kneighbors(original_referencia, return_distance=False)[0]
    original_similar = original.iloc[similar][variaveis].reset_index()
    referencia = original.iloc[similar]['referencia'].reset_index()
    referencia = referencia.merge(original_similar, on='index', how='left')
    referencia = referencia.drop(columns=['index'])
    for ind, rw in referencia.iterrows():
        if row['referencia'] != rw['referencia']:
            #print('--> {0}'.format(rw['referencia']))
            similares.insert(0, [row['referencia'], rw['referencia']])
similares = pd.DataFrame(
    similares,
    columns = ['referencia', 'vizinho']
)
similares.to_feather('../dados/knn_estado.feather')

print('Clusterização por estado...')
country_rfm = fit_data(data, 'State')
country_rfm = country_rfm.fillna(0)
country_rfm['cluster'] = KMeans(
    n_clusters=5,
    random_state=0,
    n_init='auto'
).fit(
    country_rfm[
        variaveis
    ]
).labels_
cluster = []
for index, row in enumerate(KMeans(n_clusters=5, random_state=0, n_init='auto'
    ).fit(country_rfm[variaveis]).cluster_centers_):
    cluster.insert(0,
        [index, row[0], row[1], row[2], row[3], row[4], row[5]]
    )
cluster = pd.DataFrame(
    cluster,
    columns = [
        'cluster', 'clf_vendas', 'cls_lucro',
        'clm_lucro', 'clm_qtde', 'clm_vendas', 'clr_dias'
    ]
)
country_rfm = country_rfm.merge(
    cluster,
    on='cluster',
    how='left'
)
country_rfm.to_feather('../dados/clusterizacao_estado.feather')