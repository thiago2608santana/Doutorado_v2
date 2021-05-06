import os, shutil
import numpy as np
import pandas as pd
import scipy.io as sp
import PIL.Image as pil
import re
import copy
import metodosPrincipais as mp

def mediaDosFrames(lista_frames):
    matriz_medias = []
    lista_frames = np.array(lista_frames)
    
    linhas = lista_frames[0].shape[0]
    colunas = lista_frames[0].shape[1]
    
    for i in range(linhas):
        for j in range(colunas):
            matriz_medias.append(np.mean(lista_frames[:,i,j]))
            
    matriz = np.array(matriz_medias).reshape((linhas,colunas))
    
    return matriz

#Salvar em disco as imagens convertidas para escala de cinza de acordo com os parâmetros previamente definidos
#qtdAmostras se refere à quantidade de pessoas por arquivo, ou seja, 
#voluntário controle ou diabético. qtdFrames se refere ao número de
#frames na profundidade das matrizes, ou seja, quantos frames serão salvos
#em formato de imagem.
def salvarImagens(passo, qtdAmostras, qtdFrames, dados, metaDados, diretorio_controle, diretorio_diabetico):
    lista_frames = []
    
    for k in range(len(dados)):
        for i in range(qtdAmostras):
            for j in range(0,qtdFrames+passo,passo):
                for w in range(passo):
                    frame = mp.obterFrames(dados[k]['dados_prontos'][i][0],j+w)
                    matriz = np.asarray(frame)
                    matriz[4][0] = 0
                
                    lista_frames.append(matriz)
                    
                media_dos_frames = mediaDosFrames(lista_frames)
                
                imagem = pil.fromarray(media_dos_frames, mode='L')
                
                if metaDados[k][1] == 'biceps':
                    if metaDados[k][0] == 'gc':
                        imagem.save(f'./{diretorio_controle}/{k}_{i}_{j}_controle_biceps_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                    else:
                        imagem.save(f'./{diretorio_diabetico}/{k}_{i}_{j}_diabetico_biceps_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                elif metaDados[k][1] == 'gastrocnemio':
                    if metaDados[k][0] == 'gc':
                        imagem.save(f'./{diretorio_controle}/{k}_{i}_{j}_controle_gastrocnemio_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                    else:
                        imagem.save(f'./{diretorio_diabetico}/{k}_{i}_{j}_diabetico_gastrocnemio_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                elif metaDados[k][1] == 'tibial':
                    if metaDados[k][0] == 'gc':
                        imagem.save(f'./{diretorio_controle}/{k}_{i}_{j}_controle_tibial_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                    else:
                        imagem.save(f'./{diretorio_diabetico}/{k}_{i}_{j}_diabetico_tibial_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                else:
                    if metaDados[k][0] == 'gc':
                        imagem.save(f'./{diretorio_controle}/{k}_{i}_{j}_controle_vasto_{metaDados[k][2]}_{metaDados[k][3]}.jpg')
                    else:
                        imagem.save(f'./{diretorio_diabetico}/{k}_{i}_{j}_diabetico_vasto_{metaDados[k][2]}_{metaDados[k][3]}.jpg')

#Função auxiliar que é usada como parâmetro da função sorted() para ordenar a lista de figuras em uma pasta
def ordenarImagens(element):
    element_splited = []
    element_splited.append(int(element.split('_')[0]))
    element_splited.append(int(element.split('_')[1]))
    element_splited.append(int(element.split('_')[2]))
    if element.split('_')[5] == 'max':
        element_splited.append(4)
    else:
        element_splited.append(int(element.split('_')[5]))
    element_splited.append(int(element.split('_')[6].split('.')[0]))
    return element_splited
                    
#Função que prepara os dados, criando diretórios caso não existam e copiando arquivos nos devidos lugares
def prepararDados(diretorio_base, diretorio_controle, diretorio_diabetico, qtdTrain, qtdValidation, qtdTest):
    
    itens_controle_dir = sorted(os.listdir(diretorio_controle),key=ordenarImagens)
    itens_diabetico_dir = sorted(os.listdir(diretorio_diabetico),key=ordenarImagens)
    
    mp.criarDiretorio(diretorio_base)
    
    #Criar diretórios principais de treinamento, validação e teste
    train_dir = os.path.join(diretorio_base,'train')
    mp.criarDiretorio(train_dir)
    validation_dir = os.path.join(diretorio_base,'validation')
    mp.criarDiretorio(validation_dir)
    test_dir = os.path.join(diretorio_base,'test')
    mp.criarDiretorio(test_dir)
    
    #Criar subdiretórios de treinamento, validação e teste para controle e diabético
    train_controle_dir = os.path.join(train_dir,'controle')
    mp.criarDiretorio(train_controle_dir)
    train_diabetico_dir = os.path.join(train_dir,'diabetico')
    mp.criarDiretorio(train_diabetico_dir)
    
    validation_controle_dir = os.path.join(validation_dir,'controle')
    mp.criarDiretorio(validation_controle_dir)
    validation_diabetico_dir = os.path.join(validation_dir,'diabetico')
    mp.criarDiretorio(validation_diabetico_dir)
    
    test_controle_dir = os.path.join(test_dir,'controle')
    mp.criarDiretorio(test_controle_dir)
    test_diabetico_dir = os.path.join(test_dir,'diabetico')
    mp.criarDiretorio(test_diabetico_dir)
    
    #Copiar imagens dos diretórios base para subdiretórios de treinamento, validação e teste
    for i in range(qtdTrain):
       src = os.path.join(diretorio_controle, itens_controle_dir[i])
       dst = os.path.join(train_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(diretorio_controle, itens_controle_dir[i])
       dst = os.path.join(validation_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(diretorio_controle, itens_controle_dir[i])
       dst = os.path.join(test_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain):
       src = os.path.join(diretorio_diabetico, itens_diabetico_dir[i])
       dst = os.path.join(train_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(diretorio_diabetico, itens_diabetico_dir[i])
       dst = os.path.join(validation_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)

    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(diretorio_diabetico, itens_diabetico_dir[i])
       dst = os.path.join(test_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)
       