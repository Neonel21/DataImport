
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('eksikveriler.csv')
#pd.read_csv("veriler.csv")

print(veriler)


#eksik verileri temizleme
from sklearn.impute import SimpleImputer
Yas = veriler.iloc[:,1:4].values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)

imputer = imputer.fit(Yas[:, 1:4])

Yas[:, 1:4] = imputer.transform(Yas[:, 1:4])
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:,0] = le.fit_transform(ulke[:,0])
print(ulke)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features='all')
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)
print(list(range(22)))

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'] )
print(sonuc)

sonuc2 =pd.DataFrame(data = Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = cinsiyet , index=range(22), columns=['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2= pd.concat([s,sonuc3],axis=1)
print(s2)


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print(x_train)







    
    

