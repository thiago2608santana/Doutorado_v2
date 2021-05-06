import metodosPrincipais as mp

#Definir um diretório base para armazenamento futuro dos dados de treinamento, validação e teste
base_dir = './DiretorioBase'

#Carregar os dados dos HD-sEMG
caminho = './FuzzyData/'
dados, metaDados = mp.carregarDadosDeDiretorio(caminho)

#Definir a quantidade de imagens do EMG de cada pessoa e o nome do diretório a ser criado para salvar essas imagens
qtdImagens = 500
diretorio_imagens_controle = 'ImagensGrupoControle'
diretorio_imagens_diabetico = 'ImagensGrupoDiabeticos'

#Definir a quantidade de voluntários diferentes em cada grupo
qtdAmostras = 10

#Definir a quantidade de imagens nos conjuntos de treinamento, validação e teste
qtdPercentTrain = 50
qtdPercentValidation = 30
qtdPercentTest = 20
qtdSituacoes = 1
qtdTentativas = 1
qtdTrain, qtdValidation, qtdTest = mp.definirTreinamentoValidacao(qtdPercentTrain, 
                                                                  qtdPercentValidation, 
                                                                  qtdPercentTest, 
                                                                  qtdAmostras,qtdImagens,
                                                                  qtdSituacoes,
                                                                  qtdTentativas)

mp.criarDiretorio(diretorio_imagens_controle)
mp.criarDiretorio(diretorio_imagens_diabetico)

for i in range(len(metaDados)):
    if metaDados[i][1] == 'controle':
        mp.salvarImagens(qtdAmostras,qtdImagens,dados[i],diretorio_imagens_controle,
                         metaDados[i][0],metaDados[i][1])
    else:
        mp.salvarImagens(qtdAmostras,qtdImagens,dados[i],diretorio_imagens_diabetico,
                         metaDados[i][0],metaDados[i][1])

mp.prepararDados(base_dir,f'./{diretorio_imagens_controle}',f'./{diretorio_imagens_diabetico}',
                 qtdTrain,qtdValidation,qtdTest)