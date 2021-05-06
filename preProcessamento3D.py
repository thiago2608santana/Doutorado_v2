import metodosPrincipais3D as mp3d
import numpy as np

#Criar um diretório base para armazenamento dos dados de treinamento, validação e teste
mp3d.criarDiretorio('DiretorioTreinoValidacao')

#Carregar os dados dos HD-sEMG
caminho = './DadosFiltrados/'


dados, metaDados = mp3d.carregarDadosDeDiretorio(caminho)

matriz4D_controle, target_controle, matriz4D_diabetico, target_diabetico = mp3d.criarMatrizes4D(dados, metaDados)

matriz4D_controle = matriz4D_controle[:,:,:,:5000]
matriz4D_diabetico = matriz4D_diabetico[:,:,:,:5000]

np.save('./DiretorioTreinoValidacao/matriz4D_controle',matriz4D_controle)
np.save('./DiretorioTreinoValidacao/target_controle',target_controle)
np.save('./DiretorioTreinoValidacao/matriz4D_diabetico',matriz4D_diabetico)
np.save('./DiretorioTreinoValidacao/target_diabetico',target_diabetico)

del matriz4D_controle
del target_controle
del matriz4D_diabetico
del target_diabetico
del caminho
del dados
del metaDados