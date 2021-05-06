import metodosPrincipais as mp

#Definir um diretório base para armazenamento futuro dos dados de treinamento, validação e teste
base_dir = './FuzzyBase/'

#Carregar os dados dos HD-sEMG
caminho = './FuzzyData/'
dados, metaDados = mp.carregarDadosFuzzyDeDiretorio(caminho)

#Definir a quantidade de imagens do EMG de cada pessoa e o nome do diretório a ser criado para salvar essas imagens
qtdImagens = 1
diretorio_fuzzy_controle = 'ImagensFuzzyControle'
diretorio_fuzzy_diabetico = 'ImagensFuzzyDiabeticos'

#Definir a quantidade de voluntários diferentes em cada grupo
qtdAmostras = 10

#Definir a quantidade de imagens nos conjuntos de treinamento, validação e teste
qtdPercentTrain = 50
qtdPercentValidation = 30
qtdPercentTest = 20
qtdCondicoes = 4
qtdRepeticoes = 2
qtdTrain, qtdValidation, qtdTest = mp.definirTreinamentoValidacao(qtdPercentTrain, 
                                                                  qtdPercentValidation, 
                                                                  qtdPercentTest, 
                                                                  qtdAmostras,qtdImagens,
                                                                  qtdCondicoes,
                                                                  qtdRepeticoes)

mp.criarDiretorio(diretorio_fuzzy_controle)
mp.criarDiretorio(diretorio_fuzzy_diabetico)

for i in range(len(metaDados)):
    if metaDados[i][1] == 'controle':
        mp.salvarImagensFuzzy(qtdAmostras,dados[i],metaDados[i],diretorio_fuzzy_controle)
    else:
        mp.salvarImagensFuzzy(qtdAmostras,dados[i],metaDados[i],diretorio_fuzzy_diabetico)

mp.prepararDados(base_dir,f'./{diretorio_fuzzy_controle}',f'./{diretorio_fuzzy_diabetico}',
                 qtdTrain,qtdValidation,qtdTest)