#Métodos para avaliação de acurácia, matriz de confusão e seus desdobramentos
import pandas as pd

#Calcula a matriz de confusão e retorna essa matriz
def calcularMatrizConfusao(matriz, valor_predito, valor_atual):
    
    TP = matriz[0,0]
    FP = matriz[0,1]
    FN = matriz[1,0]
    TN = matriz[1,1]

    if valor_predito == True and valor_atual == 1:
        TP = TP + 1
    elif valor_predito == True and valor_atual == 0:
        FP = FP + 1
    elif valor_predito == False and valor_atual == 1:
        FN = FN + 1
    else:
        TN = TN + 1

    matriz[0,0] = TP
    matriz[0,1] = FP
    matriz[1,0] = FN
    matriz[1,1] = TN

    return matriz

#Calcula os scores a partir dos resultados da matriz de confusão
def calcularScores(matriz):
    accuracy = (matriz[0,0] + matriz[1,1])/(matriz[0,0] + matriz[1,1] + matriz[0,1] + matriz[1,0])
    missclassification = (matriz[0,1] + matriz[1,0])/(matriz[0,0] + matriz[1,1] + matriz[0,1] + matriz[1,0])
    precision = matriz[0,0]/(matriz[0,0] + matriz[0,1])
    recall = matriz[0,0]/(matriz[0,0] + matriz[1,0])
    specificity = matriz[1,1]/(matriz[1,1] + matriz[0,1])
    f1_score = (2*matriz[0,0])/(2*matriz[0,0] + matriz[0,1] + matriz[1,0])
    csi = matriz[0,0]/(matriz[0,0] + matriz[0,1] + matriz[1,0])

    return [accuracy, missclassification, precision, recall, specificity, f1_score, csi]

#Salva todos os scores calculados a partir da matriz de confusão em um arquivo
#excel
def salvarScoresExcel(lista_scores,caminho):
      
    dicionario_scores = {'Accuracy': lista_scores[0], 'Missclassification': lista_scores[1], 'Precision': lista_scores[2], 'Recall': lista_scores[3], 'Specificity': lista_scores[4], 'F1_score': lista_scores[5], 'CSI': lista_scores[6]}
    
    df_scores = pd.DataFrame(dicionario_scores, index=[0])
    
    df_scores.to_excel(f'{caminho}scores_fuzzy_cnn2D.xlsx', index=False)

