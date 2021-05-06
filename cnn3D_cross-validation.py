import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import KFold
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
#matriz4D_controle = np.float32(matriz4D_controle)
target_controle = np.load(f'{caminho}target_controle.npy')
matriz4D_diabetico = np.load(f'{caminho}matriz4D_diabetico.npy')
#matriz4D_diabetico = np.float32(matriz4D_diabetico)
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

indice = 1
VALIDATION_ACCURACY = []
VALIDATION_LOSS = []

with tf.device('/GPU:0'):
    
    X_train = np.reshape(X_train,(len(X_train),5,12,5000,1))
    X_test = np.reshape(X_test,(len(X_test),5,12,5000,1))
    
    sample_shape = (5, 12, 5000, 1)
    
    #Definição do modelo
    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=(2, 2, 3000), activation='relu', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 10)))
    #model.add(Conv3D(filters=64, kernel_size=(2, 2, 2), activation='relu', padding='same'))
    #model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    kf = KFold(n_splits=5)
    
    for train_index, val_index in kf.split(X_train, y_train):
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(caminho_salvar+mp3d.obterNomeDoModelo(indice)
                                                        ,monitor='accuracy'
                                                        ,save_best_only=True
                                                        ,mode='max'
                                                        ,save_freq='epoch')
        
        callbacks_list = [checkpoint]
        
        history = model.fit(X_train[train_index], y_train[train_index], batch_size=10, epochs=30, callbacks=callbacks_list)
        
        model.load_weights("./ResultadosCnn3D/model_"+str(indice)+".h5")
        
        results = model.evaluate(X_test)
        results = dict(zip(model.metrics_names,results))
        
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])
        
        tf.keras.backend.clear_session()
        
        indice += 1

print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")