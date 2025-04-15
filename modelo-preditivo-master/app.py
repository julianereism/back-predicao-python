from flask import Flask, render_template, request
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Inicializar Firebase
cred = credentials.Certificate("dados-para-predicao-firebase-adminsdk-fbsvc-767d0a65cb.json.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)

# Carregar modelo
model = pickle.load(open('modelo_preditivo_random_forest.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Coletar dados do formulário
        compra_online = int(request.form.get("compra-online"))
        distancia_casa = float(request.form.get("distancia-casa"))
        distancia_ultima_transacao = int(request.form.get("distancia-ultima-transacao"))
        loja_repetida = int(request.form.get("loja-repetida"))
        razao_media_compras = float(request.form.get("razao-media-compras"))
        uso_chip = int(request.form.get("uso-chip"))
        uso_codigo_seguranca = int(request.form.get("uso-codigo-seguranca"))
        fraude = int(request.form.get("fraude"))  # Esse campo é geralmente o target, mas vamos armazenar por enquanto
        cidade = request.form.get("cidade")
        bairro = request.form.get("bairro")

        # Construir o array de features (exceto cidade/bairro que são apenas metadados)
        features = np.array([[compra_online, distancia_casa, distancia_ultima_transacao,
                              loja_repetida, razao_media_compras, uso_chip,
                              uso_codigo_seguranca, fraude]])

        previsao = model.predict(features)[0]

        # Enviar dados para o Firebase
        dados = {
            "compra_online": compra_online,
            "distancia_casa": distancia_casa,
            "distancia_ultima_transacao": distancia_ultima_transacao,
            "loja_repetida": loja_repetida,
            "razao_media_compras": razao_media_compras,
            "uso_chip": uso_chip,
            "uso_codigo_seguranca": uso_codigo_seguranca,
            "fraude": fraude,
            "cidade": cidade,
            "bairro": bairro,
            "risco_fraude_previsto": float(previsao)
        }

        db.collection("transacoes_fraude").add(dados)

        return render_template("index.html", previsao=previsao)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
