import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import metodosPrincipais3D as mp3d
#import metodosValidacaoResultados as mvr
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import time
from win10toast import ToastNotifier

#Criar um objeto para notificação
toast = ToastNotifier()

#Iniciar contador de tempo
start_time = time.time()

#Exibir a GPU disponível
device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))

#Diretorio para salvar os resultados
caminho_salvar = './ResultadosCnn3D/'

#Carregar os dados dos HD-sEMG já em formato 3D
caminho = './DiretorioTreinoValidacao/'

matriz4D_controle = np.load(f'{caminho}matriz4D_controle.npy')
matriz4D_controle = np.float32(matriz4D_controle)
target_controle = np.load(f'{caminho}target_controle.npy')
matriz4D_diabetico = np.load(f'{caminho}matriz4D_diabetico.npy')
matriz4D_diabetico = np.float32(matriz4D_diabetico)
target_diabetico = np.load(f'{caminho}target_diabetico.npy')

#Igualar os conjuntos de dados CONTROLE e DIABETICO. Excluir metade dos
#diabéticos.
indices_diabetico = mp3d.obterIndicesTreinamentoDiabeticos(len(matriz4D_diabetico))
matriz4D_diabetico = matriz4D_diabetico[indices_diabetico]
target_diabetico = target_diabetico[indices_diabetico]

#Agregar os dados em um único conjunto para o split de treino e teste
X, y = mp3d.criarConjuntoTreinamento(matriz4D_controle, matriz4D_diabetico, target_controle, target_diabetico)

del matriz4D_controle
del matriz4D_diabetico
del target_controle
del target_diabetico

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

with tf.device('/GPU:0'):
    
    X_train = np.reshape(X_train,(len(X_train),5,12,12000,1))
    X_test = np.reshape(X_test,(len(X_test),5,12,12000,1))
    
    sample_shape = (5, 12, 12000, 1)
    
    #Definição do modelo
    model = Sequential()
    model.add(Conv3D(filters=32, kernel_size=(2, 2, 3000), activation='relu', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 10)))
    #model.add(Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()

    model.fit(X_train, y_train, batch_size=10, epochs=30)
    
    predicao = model.predict(X_test)
    
    predicao = np.array(predicao > 0.5)
        
    matriz = confusion_matrix(y_test, predicao)
    
    #probabilidades = model.predict_proba(X_test)
    
    #fpr, tpr, thresholds = roc_curve(y_test,probabilidades)
    #roc_auc = auc(fpr, tpr)

#lista_scores = mvr.calcularScores(matriz)
#mvr.salvarScoresExcel(lista_scores,caminho_salvar)

#plt.figure(1, figsize=(12,6))
#plt.plot(fpr, tpr, color='r', lw=2, alpha=0.5, label='ROC (AUC = %0.2f)' % (roc_auc))
#plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.grid()
#plt.show()
#plt.savefig('curva_roc_cnn3D.png', dpi=400)

print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")