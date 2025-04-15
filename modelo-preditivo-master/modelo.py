import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Gerar dados fictícios (exemplo)
dados = {
    'compra_online': np.random.randint(0, 2, size=100),
    'distancia_casa': np.random.uniform(0.1, 100.0, size=100),
    'distancia_ultima_transacao': np.random.randint(1, 50, size=100),
    'loja_repetida': np.random.randint(0, 2, size=100),
    'razao_media_compras': np.random.uniform(0.1, 5.0, size=100),
    'uso_chip': np.random.randint(0, 2, size=100),
    'uso_codigo_seguranca': np.random.randint(0, 2, size=100),
    'fraude': np.random.randint(0, 2, size=100)  # classe alvo
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Separar features e alvo
X = df.drop('fraude', axis=1)
y = df['fraude']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliar o modelo
predicoes = modelo.predict(X_test)
print(classification_report(y_test, predicoes))

# Salvar modelo
with open('modelo_preditivo_random_forest.pkl', 'wb') as f:
    pickle.dump(modelo, f)
