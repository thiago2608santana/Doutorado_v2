from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import metodosPrincipais as mp
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

caminho_teste = './FeaturesBase/all_test/'

modelo = models.load_model('./FeaturesBase/model_5.h5')
modelo.summary()

modelo_intermediario = Model(inputs=modelo.input, outputs=modelo.get_layer('dense_1').output)
modelo_intermediario.summary()

dados, etiquetas = mp.criarMatrizes3DDoDiretorio(caminho_teste)

features = modelo_intermediario.predict(dados)

dados_treinamento = pd.DataFrame(features)
dados_treinamento['target'] = etiquetas

##############################################################################
##########################Seleção de características##########################
##############################################################################

numero_splits = 5

gb = GradientBoostingClassifier()
rfecv = RFECV(estimator=gb, step=1, cv=StratifiedKFold(numero_splits), scoring='accuracy')
rfecv.fit(dados_treinamento.drop('target', axis=1), dados_treinamento['target'])

selected = dados_treinamento[np.array(dados_treinamento.columns[np.where(rfecv.support_ == True)[0]])] 
selected['target'] = dados_treinamento['target']

###Machine Learning###
X_train, X_test, y_train, y_test = train_test_split(selected.drop('target', axis=1), selected['target'], test_size=0.30)

lr = LogisticRegression()
rf = RandomForestClassifier()
knn_model = KNeighborsClassifier()

###Logistic Regression###
lr.fit(X_train,y_train)
predicao_lr = lr.predict(X_test)

###Random Forest###
rf.fit(X_train,y_train)
predicao_rf = rf.predict(X_test)

###KNN###
knn_model.fit(X_train,y_train)
predicao_knn = knn_model.predict(X_test)

print('Métricas Regressão Logística')
print(confusion_matrix(y_test,predicao_lr))
print(classification_report(y_test,predicao_lr))

print('Métricas Floresta Aleatória')
print(confusion_matrix(y_test,predicao_rf))
print(classification_report(y_test,predicao_rf))

print('Métricas KNN')
print(confusion_matrix(y_test,predicao_knn))
print(classification_report(y_test,predicao_knn))