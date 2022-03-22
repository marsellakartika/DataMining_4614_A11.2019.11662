import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

#Mengimport data
datmin = pd.read_csv('data.csv',sep=";")
datmin.head(5)

#Mengidentifikasi nilai tak wajar sebagai missing value

datmin['Glukosa']=datmin['Glukosa'].replace(0,np.nan)
datmin['TekananDarah']=datmin['TekananDarah'].replace(0,np.nan)
datmin['KetebalanKulit']=datmin['KetebalanKulit'].replace(0,np.nan)
datmin['Insulin']=datmin['Insulin'].replace(0,np.nan)
datmin['BMI']=datmin['BMI'].replace(0,np.nan)
datmin.head()

#imputasi class mean pada attribute Glukosa dan Insulin
df1['Glukosa'].fillna(df1['Glukosa'].mean(),inplace=True)
df2['Glukosa'].fillna(df2['Glukosa'].mean(),inplace=True)

df1['Insulin'].fillna(df1['Insulin'].median(),inplace=True)
df2['Insulin'].fillna(df2['Insulin'].median(),inplace=True)
datmin2=df1.append(df2)
datmin2.head()

#Imputasi Mean pada tekanan darah,ketebalan tubuh, dan BMI
mean1=datmin2['TekananDarah'].mean()
datmin2['TekananDarah'].fillna(mean1,inplace=True)
mean2=datmin2['KetebalanKulit'].mean()
datmin2['KetebalanKulit'].fillna(mean2,inplace=True)
mean3=datmin2['BMI'].mean()
datmin2['BMI'].fillna(mean3,inplace=True)

datmin2.describe().transpose()

#Missing Value
total=datmin2.isnull().sum().sort_values(ascending = False)
print(total)

#Cek outlier menggunakan boxplot
datat=['Kehamilan','Glukosa','TekananDarah','KetebalanKulit','Insulin','BMI','DiabetesPedigreeFunction','Age']
ax = sns.boxplot(data=datmin2[datat], orient="h", palette="Set2")
ax = sns.boxplot(data=datscale[datat], orient="h", palette="Set2")

#mengambil data tanpa kolom target
datout2=datmin2[['Kehamilan','Glukosa','TekananDarah',
                'KetebalanKulit','Insulin','BMI','DiabetesPedigreeFunction','Age']]

#mengetahui banyak data (n)
df=len(list(datout2.columns.values))

#membuat fungsi mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()
datout2['mahala'] = mahalanobis(x=datout2, data=datout2)
datout2.head()

#membuat fungsi deteksi outlier MD
from scipy.stats import chi2
def MD_detectOutliers(MD,df):
    nilaichi=chi2.isf(0.01, df)
    outliers = []
    for i in range(len(MD)):
        if (MD[i] > nilaichi):
            outliers.append(i)  
# index of the outlier
    return np.array(outliers)

outliers_indices = MD_detectOutliers(datout2['mahala'],df)

print("Outliers Indices: {}\n".format(outliers_indices))
len(outliers_indices)

d2=datout2['mahala']
eks = range( len( d2 ))

plt.subplot(111)

plt.scatter( eks, d2 )

plt.hlines( chi2.ppf(0.99, df), 0, len(d2), label ="99% $\chi^2$ quantile", linestyles = "solid" )  

plt.legend()
plt.ylabel("recorded value")
plt.xlabel("observation")
plt.title( 'Mahalanobis detection of outliers at 99% $\chi^2$ quantiles' )

plt.show()


