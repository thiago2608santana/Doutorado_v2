from tensorflow.python.keras import models
from tensorflow.python.keras import Model
from tensorflow.python.keras.preprocessing.image import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from feature_selector import FeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import metodosPrincipais as mp
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

caminho_teste = './FeaturesBase/all_test/'

#modelo = models.load_model('./feature_extraction_model.h5')
modelo = models.load_model('./FeaturesBase/model_5.h5')
modelo.summary()

modelo_intermediario = Model(inputs=modelo.input, outputs=modelo.get_layer('dense_1').output)
modelo_intermediario.summary()

#dados = image_dataset_from_directory(directory=caminho_teste,
#                                     labels='inferred',
#                                     label_mode=None,
#                                     #class_names=['controle','diabetico'],
#                                     color_mode='grayscale',
#                                     image_size=(12, 5))

dados, etiquetas = mp.criarMatrizes3DDoDiretorio(caminho_teste)

features = modelo_intermediario.predict(dados)

#features_controle = np.zeros((len(features)//2,1))
#features_diabetico = np.ones((len(features)//2,1))
#targets = np.concatenate((features_controle, features_diabetico))
#targets = targets.astype(int)
#targets = targets.reshape(-1)

dados_treinamento = pd.DataFrame(features)
dados_treinamento['target'] = etiquetas

##############################################################################
##########################Seleção de características##########################
##############################################################################
fs = FeatureSelector(data=dados_treinamento, labels=dados_treinamento['target'])

###Identificar valores faltantes###
fs.identify_missing(missing_threshold=0.6)

###Identificar valores únicos###
fs.identify_single_unique()
single_unique = fs.ops['single_unique']

###Identificar características colineares (altamente correlacionadas)###
fs.identify_collinear(correlation_threshold=0.975)
correlated_features = fs.ops['collinear']

###Identificar características com zero importância###
fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', 
                            n_iterations = 10, early_stopping = True)
zero_importance_features = fs.ops['zero_importance']

###Identificar importância baixa###
fs.identify_low_importance(cumulative_importance = 0.99)
low_importance_features = fs.ops['low_importance']

###Remover características sem importância###

#Verificar todas as características a serem removidas
all_to_remove = fs.check_removal()
#Remover
dados_treinamento_final = fs.remove(methods = 'all')

###Machine Learning###
X_train, X_test, y_train, y_test = train_test_split(dados_treinamento_final.drop('target', axis=1), dados_treinamento_final['target'], test_size=0.30)

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