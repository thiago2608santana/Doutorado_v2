import os
import numpy as np
import scipy.io as sp
import re

#Função que cria um diretório para salvar os dados caso não exista
def criarDiretorio(nomeDiretorio):
    if not os.path.isdir(f'./{nomeDiretorio}'):
        os.mkdir(f'./{nomeDiretorio}')

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

#Cria matrizes 4D para os dados do grupo controle e grupo diabético
#Devido à inconsistência de tamanho, os dados foram todos cortados em
#12000 frames. Dessa forma resultando em amostras (5,12,12000) cada.
def criarMatrizes4D(dados, metaDados):
    
    #matriz4D_controle = np.zeros((320, 5, 12, 12000),dtype='uint8')
    #matriz4D_diabetico = np.zeros((629, 5, 12, 12000),dtype='uint8')
    matriz4D_controle = np.zeros((320, 5, 12, 12000))
    matriz4D_diabetico = np.zeros((629, 5, 12, 12000))
    
    indice_controle = 0
    indice_diabetico = 0
    
    target_controle = []
    target_diabetico = []
    
    for i in range(len(metaDados)):
        if metaDados[i][0] == 'gc':
            for j in range(len(dados[i]['dados_prontos'])):
                matriz4D_controle[indice_controle] = dados[i]['dados_prontos'][j][0][:,:,:12000]
                target_controle.append(0)
                indice_controle += 1
        else:
            for j in range(len(dados[i]['dados_prontos'])):
                matriz4D_diabetico[indice_diabetico] = dados[i]['dados_prontos'][j][0][:,:,:12000]
                target_diabetico.append(1)
                indice_diabetico += 1
                
    return matriz4D_controle, target_controle, matriz4D_diabetico, target_diabetico

#Função que cria 320 índices aleatórios para escolha de quais dados do grupo
#diabético farão parte do treinamento e validação. (O grupo diabético
#contém 629 amostras, enquanto o grupo controle possui 320)
def obterIndicesTreinamentoDiabeticos(tamanho_dados_entrada):
    
    np.random.seed(1)
    
    indices = np.random.randint(tamanho_dados_entrada, size=320)
    
    return indices

#Função que junta os conjuntos de dados CONTROLE e DIABETICO e retorna
#um único juntamente com os respectivos targets
def criarConjuntoTreinamento(matriz_controle, matriz_diabetico, target_controle, target_diabetico):
    
    X = np.zeros((640,5,12,5000))
    y = np.zeros(640)
    
    X[:320] = matriz_controle
    X[320:] = matriz_diabetico
    
    y[:320] = target_controle
    y[320:] = target_diabetico
    
    return X, y

#Função que retorna um nome para o modelo dentro de um laço kfold
def obterNomeDoModelo(indice):
    return 'model_'+str(indice)+'.h5'