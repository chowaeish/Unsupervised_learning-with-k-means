import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime as dt

from helpers.data_prep import *
from helpers.eda import *

df_ = pd.read_excel("datasets/house_prices/online_retail_II.xlsx", sheet_name= "Year 2010-2011")

df = df_.copy()

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

check_df(df)

df.dropna(inplace=True)

df.isnull().any()

# faturalardaki C iptali temsil etmektedir. İptal işlemleri çıkartıyoruz.
df = df[~df["Invoice"].str.contains("C", na=False)]

# toplam kazanç
df["TotalPrice"] = df["Quantity"] * df["Price"]

# RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)

# Recency(Yenilik) : Müşterinin son satın almasından bugüne kadar olan geçen süre
# Frequency(Sıklık) : Toplam satın alma sayısı
# Monetary( Parasal değer) : Müşterinin yaptığı toplam harcama

df["InvoiceDate"].max()

today_date = dt.datetime(2011, 12, 11)
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
rfm.head()
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T
rfm = rfm[rfm["monetary"] > 0]
rfm.head()

# scale işlemi

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(rfm)
df[0:5]

kmeans = KMeans()
k_fit = kmeans.fit(df)

k_fit.inertia_
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_

################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

ssd

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()


elbow.elbow_value_


####################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler = kmeans.labels_

df = pd.read_excel("datasets/house_prices/online_retail_II.xlsx",sheet_name="Year 2010-2011")

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]

df["TotalPrice"] = df["Quantity"] * df["Price"]
today_date = dt.datetime(2011, 12, 11)
df= df.groupby('Customer ID').agg({'InvoiceDate': lambda x: (today_date - x.max()).days,'Invoice': lambda y: y.nunique(),'TotalPrice': lambda z: z.sum()})

df=df.reset_index()
df.columns = ["Customer ID",'recency', 'frequency', 'monetary']
df = df[(df["monetary"]) > 0]

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
kumeler = kmeans.labels_
df["cluster_no"] = kumeler
df.head()
