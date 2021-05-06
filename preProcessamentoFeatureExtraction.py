import os
import metodosPrincipais as mp
import metodosPrincipaisFeatureExtraction as mpfe

#Definir um diretório base para armazenamento futuro dos dados de treinamento, validação e teste
nome_diretorio_base = 'FeaturesBase'

#Carregar os dados dos HD-sEMG
caminho = './DadosFiltrados/'
dados, metaDados = mp.carregarDadosDeDiretorio(caminho)

#Definir os nomes dos diretórios a serem criados para salvar as imagens
nome_diretorio_controle = 'ImagensControle'
nome_diretorio_diabetico = 'ImagensDiabeticos'

mp.criarDiretorio(nome_diretorio_base)
mp.criarDiretorio(nome_diretorio_controle)
mp.criarDiretorio(nome_diretorio_diabetico)

#Definição de parâmetros para salvar imagens
qtdAmostras = 10
qtdFrames = 11000

qtdDadosNoDiretorio = len(os.listdir(f'./{nome_diretorio_controle}'))

qtdTrain = round(qtdDadosNoDiretorio*0.70)
qtdValidation = round(qtdDadosNoDiretorio*0.20)
qtdTest = round(qtdDadosNoDiretorio*0.10)
passo = 500

mpfe.salvarImagens(passo,qtdAmostras,qtdFrames,dados,metaDados,nome_diretorio_controle,nome_diretorio_diabetico)

mp.prepararDados(nome_diretorio_base,nome_diretorio_controle,nome_diretorio_diabetico,qtdTrain,qtdValidation,qtdTest)
