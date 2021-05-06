import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import test
from tensorflow import device
import metodosPrincipais as mp
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

#########Para a primeira execução é necessário criar as etiquetas#############
caminho_imagens = './FeaturesBase/all/'
caminho_salvar = './FeaturesBase/'
#nome_arquivo = 'ControleDiabetico'
#mp.criarEtiquetas(caminho_imagens,caminho_salvar,nome_arquivo)
##############################################################################

train_data = pd.read_csv('./FeaturesBase/targetsControleDiabetico.csv')
#Estratégia para embaralhar a ordem dos itens no DataFrame, resetando
#os índices e jogando fora uma nova coluna criada com os índices antigos
train = train_data.sample(frac=1).reset_index(drop=True)
y = train[['etiqueta']]

kf = KFold(n_splits = 5)

lista_de_acuracias = []

with device('/GPU:0'):
    
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []
    
    all_targets = []
    all_preds = []
    all_folds = []
    
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.01)
    #Definição do modelo
    model = models.Sequential()
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same', input_shape=(5, 12, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=64, kernel_size=(2, 2), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units=128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(units=100, activation='relu'))
    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=otimizador, metrics=['accuracy'])
    model.summary()
    
    train_datagen = ImageDataGenerator(rescale=1./255)
    fold_var = 1
    
    for train_index, val_index in kf.split(np.zeros(len(y)),y):
        training_data = train.iloc[train_index]
        validation_data = train.iloc[val_index]
        
        train_data_generator = train_datagen.flow_from_dataframe(training_data, directory = caminho_imagens, 
                                                                 x_col = 'nome_arquivo', y_col = 'etiqueta', 
                                                                 class_mode = 'binary', color_mode='grayscale', 
                                                                 target_size=(5,12), shuffle = True)
        valid_data_generator = train_datagen.flow_from_dataframe(validation_data, directory = caminho_imagens, 
                                                                 x_col = 'nome_arquivo', y_col = 'etiqueta', 
                                                                 class_mode = 'binary', color_mode='grayscale', 
                                                                 target_size=(5,12), shuffle = False)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(caminho_salvar+mp.obterNomeDoModelo(fold_var), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
        
        callbacks_list = [checkpoint]
        
        history = model.fit(train_data_generator, epochs=20, 
                            validation_data=valid_data_generator, callbacks=callbacks_list)
        
        model.load_weights(f'{caminho_salvar}model_{fold_var}.h5')
        
        all_targets.append(valid_data_generator.classes)
        all_preds.append(model.predict(valid_data_generator))
        all_folds.append(fold_var)
        
        results = model.evaluate(valid_data_generator)
        results = dict(zip(model.metrics_names,results))
        
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])
        
        lista_de_acuracias.append(VALIDATION_ACCURACY)
        
        tf.keras.backend.clear_session()
        
        fold_var += 1

for i in range(5):
    fpr, tpr, thresholds = roc_curve(np.array(all_targets[i]),all_preds[i])
    roc_auc = auc(fpr, tpr)
    plt.figure(1, figsize=(12,6))
    plt.plot(fpr, tpr, color='r', lw=2, alpha=0.5, label='5-FOLD C-V ROC (AUC = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='Chance level', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - 5-Fold Cross-Validation')
    plt.legend(loc="lower right")
    plt.grid()
    #plt.show()
    plt.savefig(f'{i}_curva_roc_cnn2D_5-Fold.png', dpi=400)
    plt.clf()
    
    predicao = (all_preds[i] > 0.5)
    print(f'Matriz de confusão {i}')
    print(confusion_matrix(np.array(all_targets[i]),predicao))

acuracia_media = np.mean(lista_de_acuracias[0])
print('--- %s segundos de execução ---' % (time.time() - start_time))
toast.show_toast('Notificação','O algoritmo terminou a execução!',duration=20,icon_path="python_icone.ico")