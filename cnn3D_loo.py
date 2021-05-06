import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import LeaveOneOut
import metodosPrincipais3D as mp3d
import metodosValidacaoResultados as mvr
import matplotlib.pyplot as plt
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
target_controle = np.load(f'{caminho}target_controle.npy')
matriz4D_diabetico = np.load(f'{caminho}matriz4D_diabetico.npy')
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

with tf.device('/GPU:0'):
    matriz = np.array([[0, 0], [0, 0]])
    
    all_targets = []
    all_probs = []
    
    X = np.reshape(X,(len(X),5,12,5000,1))
    
    sample_shape = (5, 12, 5000, 1)
    
    #Definição do modelo
    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3000), activation='relu', padding='same', input_shape=sample_shape))
    model.add(MaxPooling3D(pool_size=(2, 2, 100), padding='same'))
    model.add(Conv3D(filters=128, kernel_size=(3, 3, 3000), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 100), padding='same'))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    loo = LeaveOneOut()
    
    for train_index, val_index in loo.split(X, y):
        
        #checkpoint = tf.keras.callbacks.ModelCheckpoint(caminho_salvar+mp3d.obterNomeDoModelo(indice)
        #                                                ,monitor='accuracy'
        #                                                ,save_best_only=True
        #                                                ,mode='max'
        #                                                ,save_freq='epoch')
        
        model.fit(X[train_index], y[train_index], batch_size=10, epochs=30)
        
        #Salvar valores na lista para montar a curva ROC
        all_targets.append(y[val_index])
        all_probs.append(model.predict_proba(X[val_index]))
        
        predicao = model.predict(X[val_index])
        
        predicao = np.array(predicao[0] > 0.5)
        
        matriz = mvr.calcularMatrizConfusao(matriz,predicao,y[val_index])
    
    all_targets = np.array(all_targets)
    all_targets = np.reshape(all_targets,(640,1))
    all_probs = np.array(all_probs)
    all_probs = np.reshape(all_probs, (640,1))

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
plt.title('Receiver operating characteristic - Leave-One-Out Cross-Validation - Cnn3D')
plt.legend(loc="lower right")
plt.grid()
#plt.show()
plt.savefig('curva_roc_cnn3D_loo.png', dpi=400)

print('Matriz de confusão:')
print(matriz)
print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")