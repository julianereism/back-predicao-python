import pickle
import pandas as pd
import numpy as np

# Carregar o modelo
with open('modelo_preditivo_random_forest.pkl', 'rb') as f:
    modelo_carregado = pickle.load(f)

# Definir os dados com os nomes das colunas
novos_dados = pd.DataFrame({
    'compra_online': [1],  # Simulação de compra online
    'distancia_casa': [5.0],  # Distância curta
    'distancia_ultima_transacao': [2],  # Distância de última transação
    'loja_repetida': [0],  # Loja não repetida
    'razao_media_compras': [2.0],  # Valor médio de compras
    'uso_chip': [1],  # Chip de celular utilizado
    'uso_codigo_seguranca': [1]  # Código de segurança utilizado
})


# Fazer previsão
previsao = modelo_carregado.predict(novos_dados)

# Exibir o resultado da previsão (0 = não fraude, 1 = fraude)
print("Previsão de fraude:", previsao[0])
