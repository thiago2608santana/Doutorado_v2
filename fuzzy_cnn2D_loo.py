from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
#from sklearn.model_selection import KFold
from tensorflow import test
from tensorflow import device
import metodosPrincipais as mtp
import metodosValidacaoResultados as mvr
import numpy as np
import pandas as pd
import time
from win10toast import ToastNotifier
import matplotlib.pyplot as plt

#Criar um objeto para notificação
toast = ToastNotifier()

#Iniciar contador de tempo
start_time = time.time()

device_name = test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

#Criar arquivos CSV com as etiquetas 0 e 1 para target dos grupos
#juntamente com o nome dos arquivos contidos na pasta
caminho_imagens = './FuzzyBase/train/todos/'
caminho_salvar = './FuzzyBase/'

dados = mtp.carregarImagensDoDiretorio(caminho_imagens)

etiquetas_treinamento = mtp.criarEtiquetas(caminho_imagens, caminho_salvar, '_cnn2D_fuzzy_loo')

etiquetas_treinamento = pd.read_csv(f'{caminho_salvar}targets.csv')

target = np.array(etiquetas_treinamento[['etiqueta']])

with device('/GPU:0'):
    matriz = np.array([[0, 0], [0, 0]])
    
    all_targets = []
    all_probs = []
    
    #Normalizar os dados antes de treinar
    dados_normalizados = dados.astype('float32')
    dados_normalizados = dados_normalizados / 255.0
    dados_normalizados = np.reshape(dados_normalizados,(len(dados_normalizados),5,12,1))
    
    #Definição do modelo
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(5, 12, 1)))
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=512, activation='relu'))
    model.add(layers.Dense(units=256, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    #model.summary()

    loo = LeaveOneOut()
    #kf = KFold(n_splits=5)
    
    for train_index, val_index in loo.split(np.zeros(len(target)), target):
    #for train_index, val_index in kf.split(np.zeros(len(target)), target):
        
        model.fit(dados_normalizados[train_index], target[train_index], batch_size=10, epochs=50)
        
        #Salvar valores na lista para montar a curva ROC
        all_targets.append(target[val_index])
        all_probs.append(model.predict_proba(dados_normalizados[val_index]))
        
        predicao = model.predict(dados_normalizados[val_index])
        
        predicao = np.array(predicao[0] > 0.5)
        
        matriz = mvr.calcularMatrizConfusao(matriz,predicao,target[val_index])
    
    all_targets = np.array(all_targets)
    all_targets = np.reshape(all_targets,(320,1))
    all_probs = np.array(all_probs)
    all_probs = np.reshape(all_probs, (320,1))
    
    lista_scores = mvr.calcularScores(matriz)
    mvr.salvarScoresExcel(lista_scores,caminho_salvar)

fpr, tpr, thresholds = roc_curve(all_targets,all_probs)
roc_auc = auc(fpr, tpr)
plt.figure(1, figsize=(12,6))
plt.plot(fpr, tpr, color='r', lw=2, alpha=0.5, label='LOOCV ROC (AUC = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Leave-One-Out Cross-Validation')
plt.legend(loc="lower right")
plt.grid()
#plt.show()
plt.savefig('curva_roc_fuzzy_cnn2D_loo.png', dpi=400)

print('Matriz de confusão:')
print(matriz)
print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")
