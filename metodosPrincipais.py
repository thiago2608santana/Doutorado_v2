import os, shutil, random
import numpy as np
import pandas as pd
import scipy.io as sp
import PIL.Image as pil
import re
import copy
import metodosPrincipaisFeatureExtraction as mpfe
from tensorflow.python.keras.preprocessing import image

#Função que carrega as imagens do diretório em formato array 3D
#Retorna junto um array com os alvos (classe da qual os dados pertencem)
def criarMatrizes3DDoDiretorio(caminho):
    
    imagens = os.listdir(caminho)
    random.shuffle(imagens)
    quantidade = len(imagens)
    
    lista_etiqueta = []
    
    dados = np.zeros((quantidade, 5, 12, 1))
    
    for i in range(quantidade):
        imagem = image.load_img(f'{caminho}{imagens[i]}', color_mode='grayscale')
        dados[i] = image.img_to_array(imagem)
        
        if re.search('controle',imagens[i]):
            lista_etiqueta.append(0)
        else:
            lista_etiqueta.append(1)
    
    etiquetas = np.array(lista_etiqueta)
    
    return dados, etiquetas

#Função que carrega os arquivos contidos em um diretório e retorna um
#dicionário com os dados e um dicionário com os metadados referentes ao arquivo
def carregarDadosDeDiretorio(caminho):
    dados = {}
    metaDados = []
    
    arquivos = os.listdir(caminho)
    for i in range(len(arquivos)):
        dados[i] = sp.loadmat(f'{caminho}{arquivos[i]}')
        musculo_grupo = re.search('(.*?)_(.*?)_(.*?)_(.*?).mat',arquivos[i])
        metaDados.append((musculo_grupo.group(1),musculo_grupo.group(2),
                      musculo_grupo.group(3),musculo_grupo.group(4)))
    
    return dados, metaDados

#Função que carrega as imagens de um diretório
def carregarImagensDoDiretorio(diretorio):
    data = []
    
    for fname in os.listdir(diretorio):
        pathname = os.path.join(diretorio, fname)
        imagem = pil.open(pathname)
        matriz = np.asarray(imagem)
        data.append(matriz)
        dados = np.asarray(data)
    return dados

def obterConjuntoTreinamentoValidacao(dir_train, dir_val):
    treinamento_1 = carregarImagensDoDiretorio(f'{dir_train}/controle/')
    treinamento_2 = carregarImagensDoDiretorio(f'{dir_train}/diabetico/')
    validacao_1 = carregarImagensDoDiretorio(f'{dir_val}/controle/')
    validacao_2 = carregarImagensDoDiretorio(f'{dir_val}/diabetico/')
    
    X_train = np.concatenate((treinamento_1, treinamento_2))
    X_val = np.concatenate((validacao_1, validacao_2))
    
    y_zeros = np.zeros((int(X_train.shape[0]/2), 1))
    y_ones = np.ones((int(X_train.shape[0]/2), 1))
    y_train = np.concatenate((y_zeros, y_ones))
    
    y_zeros = np.zeros((int(X_val.shape[0]/2), 1))
    y_ones = np.ones((int(X_val.shape[0]/2), 1))
    y_val = np.concatenate((y_zeros, y_ones))
    
    return X_train, X_val, y_train, y_val

#Função que cria um diretório para salvar as imagens caso não exista
def criarDiretorio(nomeDiretorio):
    if not os.path.isdir(f'./{nomeDiretorio}'):
        os.mkdir(f'./{nomeDiretorio}')

#Função para extrair imagens(frames) separadas(os) de um único voluntário
def obterFrames(dados, indice):
    matriz = []
    for i in range(5):
        linha = []
        for j in range(12):
            linha.append(dados[i][j][indice])
        matriz.append(linha)
    return matriz

#Função auxiliar que é usada como parâmetro da função sorted() para ordenar a lista de figuras em uma pasta
def takeFirstAndLast(element):
    element_splited = []
    element_splited.append(int(element.split('_')[0]))
    element_splited.append(int(element.split('_')[5].split('.')[0]))
    return element_splited

def ordenarImagens(element):
    element_splited = []
    element_splited.append(int(element.split('_')[0]))
    if element.split('_')[3] == 'max':
        element_splited.append(4)
    else:
        element_splited.append(int(element.split('_')[3]))
    element_splited.append(int(element.split('_')[4].split('.')[0]))
    return element_splited
    
#Salvar em disco as imagens convertidas para escala de cinza de acordo com os parâmetros previamente definidos
def salvarImagens(qtdAmostras, qtdImagens, dados, diretorio, metaDados):
    for i in range(qtdAmostras):
        for j in range(qtdImagens):
            frame = obterFrames(dados['dados_prontos'][i][0],j)
            matriz = np.asarray(frame)
            matriz[4][0] = 0
            imagem = pil.fromarray(matriz, mode='L')
            
            if metaDados[0] == 'biceps':
                if metaDados[1] == 'gc':
                    imagem.save(f'./{diretorio}/{i}_controle_biceps_{metaDados[2]}_{metaDados[3]}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_diabetico_biceps_{metaDados[2]}_{metaDados[3]}.jpg')
            elif metaDados[0] == 'gastrocnemio':
                if metaDados[1] == 'gc':
                    imagem.save(f'./{diretorio}/{i}_controle_gastrocnemio_{metaDados[2]}_{metaDados[3]}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_diabetico_gastrocnemio_{metaDados[2]}_{metaDados[3]}.jpg')
            elif metaDados[0] == 'tibial':
                if metaDados[1] == 'gc':
                    imagem.save(f'./{diretorio}/{i}_controle_tibial_{metaDados[2]}_{metaDados[3]}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_diabetico_tibial_{metaDados[2]}_{metaDados[3]}.jpg')
            else:
                if metaDados[1] == 'gc':
                    imagem.save(f'./{diretorio}/{i}_controle_vasto_{metaDados[2]}_{metaDados[3]}.jpg')
                else:
                    imagem.save(f'./{diretorio}/{i}_diabetico_vasto_{metaDados[2]}_{metaDados[3]}.jpg')


#Função que define a quantidade de amostras nos grupos de treino, validação e teste
def definirTreinamentoValidacao(qtdPercentTrain, qtdPercentValidation, 
                                qtdPercentTest, qtdAmostrasPorGrupo, qtdImagens,
                                qtdCondicoes, qtdRepeticoes):
    qtdAmostras = 2*qtdAmostrasPorGrupo
    train = round((qtdPercentTrain*qtdAmostras)/100)
    validation = round((qtdPercentValidation*qtdAmostras)/100)
    test = round((qtdPercentTest*qtdAmostras)/100)
    
    qtdTrain = 2*qtdCondicoes*qtdRepeticoes*train*qtdImagens
    qtdValidation = 2*qtdCondicoes*qtdRepeticoes*validation*qtdImagens
    qtdTest = 2*qtdCondicoes*qtdRepeticoes*test*qtdImagens
    
    return qtdTrain, qtdValidation, qtdTest
    
#Função que prepara os dados, criando diretórios caso não existam e copiando arquivos nos devidos lugares
def prepararDados(base_dir, controle_dir, diabetico_dir, qtdTrain, qtdValidation, qtdTest):
    #itens_controle_dir = sorted(os.listdir(controle_dir),key=takeFirstAndLast)
    #itens_diabetico_dir = sorted(os.listdir(diabetico_dir),key=takeFirstAndLast)
    
    itens_controle_dir = sorted(os.listdir(controle_dir),key=mpfe.ordenarImagens)
    itens_diabetico_dir = sorted(os.listdir(diabetico_dir),key=mpfe.ordenarImagens)
    
    criarDiretorio(base_dir)
    
    #Criar diretórios principais de treinamento, validação e teste
    train_dir = os.path.join(base_dir,'train')
    criarDiretorio(train_dir)
    validation_dir = os.path.join(base_dir,'validation')
    criarDiretorio(validation_dir)
    test_dir = os.path.join(base_dir,'test')
    criarDiretorio(test_dir)
    
    #Criar subdiretórios de treinamento, validação e teste para controle e diabético
    train_controle_dir = os.path.join(train_dir,'controle')
    criarDiretorio(train_controle_dir)
    train_diabetico_dir = os.path.join(train_dir,'diabetico')
    criarDiretorio(train_diabetico_dir)
    
    validation_controle_dir = os.path.join(validation_dir,'controle')
    criarDiretorio(validation_controle_dir)
    validation_diabetico_dir = os.path.join(validation_dir,'diabetico')
    criarDiretorio(validation_diabetico_dir)
    
    test_controle_dir = os.path.join(test_dir,'controle')
    criarDiretorio(test_controle_dir)
    test_diabetico_dir = os.path.join(test_dir,'diabetico')
    criarDiretorio(test_diabetico_dir)
    
    #Copiar imagens dos diretórios base para subdiretórios de treinamento, validação e teste
    for i in range(qtdTrain):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(train_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(validation_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(controle_dir, itens_controle_dir[i])
       dst = os.path.join(test_controle_dir, itens_controle_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(train_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)
    
    for i in range(qtdTrain, (qtdTrain + qtdValidation)):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(validation_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)

    for i in range((qtdTrain + qtdValidation), (qtdTrain + qtdValidation + qtdTest)):
       src = os.path.join(diabetico_dir, itens_diabetico_dir[i])
       dst = os.path.join(test_diabetico_dir, itens_diabetico_dir[i])
       shutil.copyfile(src, dst)

#Função que monta o arquivo com as etiquetas (labels/targets) dos dados
#Ideal para criar um DataFrame com o(a) caminho/pasta que contém as imagens
#a serem utilizadas pelo método flow_from_dataframe
def criarEtiquetas(caminho_imagens, caminho_salvar, nome):
    arquivos = os.listdir(caminho_imagens)
    
    lista_nome_arquivo = []
    lista_etiqueta = []
    
    for i in range(len(arquivos)):
        if re.search('controle',arquivos[i]):
            lista_nome_arquivo.append(arquivos[i])
            lista_etiqueta.append('controle')
            #lista_etiqueta.append(0)
        else:
            lista_nome_arquivo.append(arquivos[i])
            lista_etiqueta.append('diabetico')
            #lista_etiqueta.append(1)
        
    dicionario = {'nome_arquivo': lista_nome_arquivo, 'etiqueta': lista_etiqueta}
        
    data_frame_targets = pd.DataFrame(dicionario)
    
    data_frame_targets.to_csv(f'{caminho_salvar}/targets{nome}.csv', index=False)

#Obtém o nome do modelo dentro de um laço de validação cruzada (KFold Cross Validation)
#Ideal para que uma vez salvo o melhor modelo, ele pode ser carregado novamente.
def obterNomeDoModelo(k):
    return 'model_'+str(k)+'.h5'
#------------------------------------------------------------------------------
#Métodos para tratamento dos dados processados utilizando lógica fuzzy
#------------------------------------------------------------------------------

#Função que determina os índices de treinamento e teste para realizar a 
#validação cruzada
def validacaoCruzada(caminho_controle, caminho_diabeticos, k_fold, 
                     qtdAmostrasPorGrupo, qtdTrain, qtdValidation,
                     qtdCondicoes, qtdRepeticoes, qtdMusculos):
    id_controle = []
    id_diabeticos = []
    indices = np.array(np.arange(0, 2*qtdAmostrasPorGrupo))
    indices_treinamento = {i: None for i in range(k_fold)}
    indices_validacao = {i: None for i in range(k_fold)}
    
    arquivos_controle = os.listdir(caminho_controle)
    arquivos_diabeticos = os.listdir(caminho_diabeticos)
    
    for i in range(qtdAmostrasPorGrupo*qtdCondicoes*qtdRepeticoes*qtdMusculos):
        id_controle.append(int(arquivos_controle[i].split('_')[0]))
        id_diabeticos.append(int(arquivos_diabeticos[i].split('_')[0]))
    
    id_controle = np.asarray(id_controle)
    id_diabeticos = np.asarray(id_diabeticos)
    
    for key in range(k_fold):
        embaralhados = copy.deepcopy(indices)
        np.random.seed(key)
        np.random.shuffle(embaralhados)
        indices_treinamento[key] = embaralhados[0:qtdTrain]
        indices_validacao[key] = embaralhados[qtdTrain:qtdTrain+qtdValidation]
    
    return indices_treinamento, indices_validacao

#Função que carrega os arquivos contidos em um diretório e retorna um
#dicionário com os dados e um dicionário com os metadados referentes ao arquivo
def carregarDadosFuzzyDeDiretorio(caminho):
    dados = {}
    metaDados = []
    
    arquivos = os.listdir(caminho)
    for i in range(len(arquivos)):
        dados[i] = sp.loadmat(f'{caminho}{arquivos[i]}')
        musculo_grupo = re.search('fuzzy_(.*?)_(.*?)_(.*?)_(.*?).mat',arquivos[i])
        if musculo_grupo.group(3) == 'Max':
            metaDados.append((musculo_grupo.group(1),musculo_grupo.group(2),
                      40,musculo_grupo.group(4)))
        else:
            metaDados.append((musculo_grupo.group(1),musculo_grupo.group(2),
                      musculo_grupo.group(3),musculo_grupo.group(4)))
    
    return dados, metaDados

#Rearranjar os dados para o formato de matriz
def organizarDadosFuzzy(dados, indice):
    vetorFuzzy = dados['FuzEn'][indice][0][0]
    matrizFuzzy = np.reshape(vetorFuzzy, (5, 12))
    return matrizFuzzy

def salvarImagensFuzzy(qtdAmostras, dados, metaDados, diretorio):
    for i in range(qtdAmostras):
        matriz = organizarDadosFuzzy(dados,i)
        imagem = pil.fromarray(matriz, mode='L')
        
        if metaDados[0] == 'biceps':
            if metaDados[1] == 'controle':
                imagem.save(f'./{diretorio}/{i}_controle_fuzzy_biceps_{metaDados[2]}_{metaDados[3]}.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_diabetico_fuzzy_biceps_{metaDados[2]}_{metaDados[3]}.jpg')
        elif metaDados[0] == 'gastrocnemio':
            if metaDados[1] == 'controle':
                imagem.save(f'./{diretorio}/{i}_controle_fuzzy_gastrocnemio_{metaDados[2]}_{metaDados[3]}.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_diabetico_fuzzy_gastrocnemio_{metaDados[2]}_{metaDados[3]}.jpg')
        elif metaDados[0] == 'tibial':
            if metaDados[1] == 'controle':
                imagem.save(f'./{diretorio}/{i}_controle_fuzzy_tibial_{metaDados[2]}_{metaDados[3]}.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_diabetico_fuzzy_tibial_{metaDados[2]}_{metaDados[3]}.jpg')
        else:
            if metaDados[1] == 'controle':
                imagem.save(f'./{diretorio}/{i}_controle_fuzzy_vasto_{metaDados[2]}_{metaDados[3]}.jpg')
            else:
                imagem.save(f'./{diretorio}/{i}_diabetico_fuzzy_vasto_{metaDados[2]}_{metaDados[3]}.jpg')