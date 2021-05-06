from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow import test
from tensorflow import device
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import metodosPrincipais as mtp
import metodosValidacaoResultados as mvr
#from tensorflow.python.keras import regularizers
#from tensorflow.python.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from win10toast import ToastNotifier

#Criar um objeto para notificação
toast = ToastNotifier()

#Iniciar contador de tempo
start_time = time.time()

#Exibir a GPU disponível
device_name = test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

#Criar arquivos CSV com as etiquetas 0 e 1 para target dos grupos
#juntamente com o nome dos arquivos contidos na pasta
caminho_imagens_treinamento = './FuzzyBase/train/todos/'
caminho_imagens_validacao = './FuzzyBase/validation/todos/'
caminho_imagens_teste = './FuzzyBase/test/todos/'
caminho_salvar = './FuzzyBase/'

dados_treinamento = mtp.carregarImagensDoDiretorio(caminho_imagens_treinamento)
dados_validacao = mtp.carregarImagensDoDiretorio(caminho_imagens_validacao)
dados_teste = mtp.carregarImagensDoDiretorio(caminho_imagens_teste)

etiquetas_treinamento = mtp.criarEtiquetas(caminho_imagens_treinamento, caminho_salvar, '_cnn2D_fuzzy_treinamento')
etiquetas_treinamento = pd.read_csv(f'{caminho_salvar}targets_cnn2D_fuzzy_treinamento.csv')

etiquetas_validacao = mtp.criarEtiquetas(caminho_imagens_validacao, caminho_salvar, '_cnn2D_fuzzy_validacao')
etiquetas_validacao = pd.read_csv(f'{caminho_salvar}targets_cnn2D_fuzzy_validacao.csv')

etiquetas_teste = mtp.criarEtiquetas(caminho_imagens_teste, caminho_salvar, '_cnn2D_fuzzy_teste')
etiquetas_teste = pd.read_csv(f'{caminho_salvar}targets_cnn2D_fuzzy_teste.csv')

target_treinamento = np.array(etiquetas_treinamento[['etiqueta']])
target_validacao = np.array(etiquetas_validacao[['etiqueta']])
target_teste = np.array(etiquetas_teste[['etiqueta']])

with device('/GPU:0'):
    
    #Normalizar os dados antes de treinar
    dados_treinamento_normalizados = dados_treinamento.astype('float32')
    dados_treinamento_normalizados = dados_treinamento_normalizados / 255.0
    dados_treinamento_normalizados = np.reshape(dados_treinamento_normalizados,(len(dados_treinamento_normalizados),5,12,1))
    
    dados_validacao_normalizados = dados_validacao.astype('float32')
    dados_validacao_normalizados = dados_validacao_normalizados / 255.0
    dados_validacao_normalizados = np.reshape(dados_validacao_normalizados,(len(dados_validacao_normalizados),5,12,1))
    
    dados_teste_normalizados = dados_teste.astype('float32')
    dados_teste_normalizados = dados_teste_normalizados / 255.0
    dados_teste_normalizados = np.reshape(dados_teste_normalizados,(len(dados_teste_normalizados),5,12,1))
    
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
    
    model.summary()

    model.fit(dados_treinamento_normalizados, target_treinamento, batch_size=10, epochs=50)
    
    predicao = model.predict(dados_validacao_normalizados)
    
    predicao = np.array(predicao > 0.5)
        
    matriz = confusion_matrix(target_validacao, predicao)
    
    probabilidades = model.predict_proba(dados_validacao_normalizados)
    
    fpr, tpr, thresholds = roc_curve(target_validacao,probabilidades)
    roc_auc = auc(fpr, tpr)

lista_scores = mvr.calcularScores(matriz)
mvr.salvarScoresExcel(lista_scores,caminho_salvar)

plt.figure(1, figsize=(12,6))
plt.plot(fpr, tpr, color='r', lw=2, alpha=0.5, label='ROC (AUC = %0.2f)' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.grid()
#plt.show()
plt.savefig('curva_roc_fuzzy_cnn2D.png', dpi=400)

print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")