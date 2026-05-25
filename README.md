# Predição de Evasão Escolar com Machine Learning (XGBoost)

## Sobre o Projeto

Este projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) em Engenharia de Software e tem como objetivo prever a evasão escolar utilizando técnicas de Machine Learning.

O modelo foi construído utilizando algoritmos de classificação supervisionada, com foco principal no XGBoost, além de técnicas de:
- Seleção de atributos;
- Balanceamento de classes com SMOTE;
- Normalização dos dados;
- Otimização de hiperparâmetros com GridSearchCV.

A classificação foi simplificada em:
- Dropout
- Não Dropout

## Pipeline do Modelo

O fluxo de processamento do modelo segue as seguintes etapas:

1. Carregamento e tratamento do dataset;
2. Conversão do problema para classificação binária;
3. Normalização dos dados com MinMaxScaler;
4. Seleção de atributos com RandomForest;
5. Balanceamento das classes utilizando SMOTE;
6. Otimização de hiperparâmetros com GridSearchCV;
7. Treinamento do modelo XGBoost;
8. Avaliação do desempenho.

## Tecnologias e Bibliotecas

- Python
- Pandas
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- NumPy

## Métricas Avaliadas

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- F1-Score
