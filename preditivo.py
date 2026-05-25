import pandas as pd    
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Carregar os dados
df = pd.read_csv("data.csv", sep=";")

df.columns = df.columns.str.strip()  # Remover espaços nos nomes das colunas

# Transformar o problema em binário: Dropout vs. Não Dropout
df["Target"] = df["Target"].replace({"Enrolled": "Nao_Dropout", "Graduate": "Nao_Dropout"})

# Converter a variável alvo para valores numéricos
label_encoder = LabelEncoder()
df["Target"] = label_encoder.fit_transform(df["Target"])

# Separar variáveis preditoras e alvo
X = df.drop(columns=["Target"])
y = df["Target"]

# Dividir os dados em treino (80%) e teste (20%) antes de normalizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar tanto o conjunto de treino quanto o de teste
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Normalização do conjunto de treino
X_test_scaled = scaler.transform(X_test)  # Aplicar a mesma transformação no conjunto de teste

# Selecionar as melhores features no conjunto de treino
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train_scaled, y_train)
feature_selector = SelectFromModel(rf_temp, threshold="mean")
X_train_selected = feature_selector.transform(X_train_scaled)
X_test_selected = feature_selector.transform(X_test_scaled)

# Aplicar SMOTE para balanceamento das classes
smote = SMOTE(sampling_strategy=0.8, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Criar pesos manuais para cada classe com base no desequilíbrio
target_counts = y_train_resampled.value_counts()
class_weights = {cls: max(target_counts) / count for cls, count in target_counts.items()}

# Criar os pesos de amostra para o treino
sample_weights = y_train_resampled.map(class_weights)

# Definir hiperparâmetros para otimização no XGBoost
param_grid = {
    "n_estimators": [150, 200, 250],  
    "max_depth": [3, 4, 5],  
    "learning_rate": [0.01, 0.03, 0.05],  
    "subsample": [0.6, 0.7],  
    "colsample_bytree": [0.6, 0.7],  
    "reg_lambda": [10, 20, 30],  
    "reg_alpha": [1, 3, 5]  
}

# Otimizar hiperparâmetros com Grid Search
grid_search = GridSearchCV(XGBClassifier(objective="binary:logistic"), 
                           param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred_test = best_model.predict(X_test_selected)

# Fazer previsões no conjunto de treino
y_pred_train = best_model.predict(X_train_resampled)

# Avaliar o modelo no conjunto de teste
accuracy_test = accuracy_score(y_test, y_pred_test)
balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
report_test = classification_report(y_test, y_pred_test, target_names=label_encoder.classes_)

# Avaliar o modelo no conjunto de treino
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)
balanced_accuracy_train = balanced_accuracy_score(y_train_resampled, y_pred_train)
report_train = classification_report(y_train_resampled, y_pred_train, target_names=label_encoder.classes_)

# Exibir resultados
print("Melhores hiperparâmetros:", grid_search.best_params_)
print("Acurácia no conjunto de treino:", accuracy_train)
print("Balanced Accuracy no conjunto de treino:", balanced_accuracy_train)
print("Relatório de classificação no conjunto de treino:\n", report_train)
print("Acurácia no conjunto de teste:", accuracy_test)
print("Balanced Accuracy no conjunto de teste:", balanced_accuracy_test)
print("Relatório de classificação no conjunto de teste:\n", report_test)
