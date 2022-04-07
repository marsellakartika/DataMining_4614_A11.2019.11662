import pandas as pd
import numpy as np

# input data
Cryotherapy=pd.read_excel(“nvbys.xls”)

# Menampilkan data
nvbys.head()

# menampilkan informasi data
nvbys.info(

# Mengecek apakah ada deret yang kosong
nvbys.empty

# Variabel independen
x = nvbys.drop([“Result_of_Treatment”], axis = 1)
x.head()

# Variabel dependen
y = nvbys[“Result_of_Treatment”]
y.head()

# Import train_test_split function
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
modelnb = GaussianNB()

# Memasukkan data training pada fungsi klasifikasi naive bayes
nbtrain = modelnb.fit(x_train, y_train)
nbtrain.class_count_

# Menentukan hasil prediksi dari x_test
y_pred = nbtrain.predict(x_test)
y_pred

# Menentukan probabilitas hasil prediksi
nbtrain.predict_proba(x_test)

# import confusion_matrix model
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

# Merapikan hasil confusion matrix
y_actual1 = pd.Series([1, 0,1,0,1,0,1,0,1,0,0,1,1,0,1,1,0,0], name = “actual”)
y_pred1 = pd.Series([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1], name = “prediction”)
df_confusion = pd.crosstab(y_actual1, y_pred1)
df_confusion

# Menghitung nilai akurasi dari klasifikasi naive bayes 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))