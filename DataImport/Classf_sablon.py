
#1. kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")

x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
print(y)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#sınıflandırma algoritmaları
# 1. Logistic Regression
#Bir veya birden fazla bağımsız değişkeni bulunan ve bir sonucu belirlemek için 
#kullanılan istatistik yönetimidir. 
#Var olan bir veri kümesinin analizi sonucu iki olası sonucu bize verir. 
#Doğrusal sınıflandırma problemlerinde kullanılır.
#Lojistik regresyon, ikili(binary) 1 veya 0 olarak kodlanmış verileri içerir.

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) #egitim

y_pred = logr.predict(X_test) #tahmin
print(y_pred)
print(y_test)

#karmasiklik matrisi
#Gerçekte var olanlarda , sizin tahminlediğiniz verileri bize verir 
cm = confusion_matrix(y_test,y_pred)
print(cm)
