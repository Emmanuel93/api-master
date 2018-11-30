from flask import Flask
from flask import jsonify
from optimizacion.algoritmogenetico import CompactGeneticAlgorithm

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/init',methods=['POST'])
def init():
    genetico = CompactGeneticAlgorithm(100, 19, 100)
    nombreArchivo = "wind_aristeomercado_10m_complete_21374-25374_suavizado"
    rutaArchivo = nombreArchivo+'.csv'
    serie = helper.leerArchivo(rutaArchivo)

    serie = helper.normalizarSerie(serie)

    print "serieNormailzadoa ", np.array(serie).reshape(len(serie), 1)

    mejorModelo.model.save("mejorModeloPaso"+str(numeroPaso)+".h5")

    return jsonify({ 'succes':true })