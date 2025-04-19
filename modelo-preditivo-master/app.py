from flask import Flask, jsonify, render_template, request
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from flask_cors import CORS
# Inicializar Firebase
cred = credentials.Certificate("dados-para-predicao-firebase-adminsdk-fbsvc-03619f1be6.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])
# Carregar modelo
model = pickle.load(open('modelo_preditivo_random_forest.pkl', 'rb'))

@app.route("/api/fraude", methods=["GET", "POST"])
def api_fraude():
    if request.method == "POST":
        # Coletar dados do formulário
        compra_online = int(request.form.get("compra-online"))
        distancia_casa = float(request.form.get("distancia-casa"))
        distancia_ultima_transacao = float(request.form.get("distancia-ultima-transacao"))
        loja_repetida = int(request.form.get("loja-repetida"))
        razao_media_compras = float(request.form.get("razao-media-compras"))
        uso_chip = int(request.form.get("uso-chip"))
        uso_codigo_seguranca = int(request.form.get("uso-codigo-seguranca"))
        fraude = int(request.form.get("fraude"))  # Apenas para armazenar
        cidade = request.form.get("cidade")
        bairro = request.form.get("bairro")
        data_fraude = datetime.now().strftime("%Y-%m-%d")

        # Criar vetor de features
        features = np.array([[compra_online, distancia_casa, distancia_ultima_transacao,
                              loja_repetida, razao_media_compras, uso_chip,
                              uso_codigo_seguranca]])

        # Fazer predição
        previsao = model.predict(features)[0]
        

        # Enviar para o Firestore
        dados = {
            "compra-online": compra_online,
            "distancia-casa": distancia_casa,
            "distancia-ultima-transacao": distancia_ultima_transacao,
            "loja-repetida": loja_repetida,
            "razao-media-compras": razao_media_compras,
            "uso_chip": uso_chip,
            "uso-codigo-seguranca": uso_codigo_seguranca,
            "fraude": fraude,
            "cidade": cidade,
            "bairro": bairro,
            "data_fraude": data_fraude,
        }

        db.collection("transacoes_fraude").add(dados)

        return jsonify({
            "message": "Transação registrada com sucesso.",
            "risco_fraude_previsto": float(previsao)
        }), 200


@app.route("/api/fraude/dados", methods=["GET"])
def listar_transacoes():
    docs = db.collection("transacoes_fraude").stream()
    transacoes = []

    for doc in docs:
        transacao = doc.to_dict()
        transacao['id'] = doc.id
        transacoes.append(transacao)

    return jsonify(transacoes), 200
   


if __name__ == "__main__":
    app.run(debug=True)
