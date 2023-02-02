import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from Model import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
# model = pickle.load(open('../Model/bilstm_2epochs.h5', 'rb'))

# tf_config = some_custom_config
sess = tf.Session()
graph = tf.get_default_graph()
set_session(sess)

model = load_model('../Model/bilstm_2epochs.h5')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        text = [x for x in request.form.values()][0]
        text_preprocess = preprocessing(text)
        text_tokenize = tokenize([text_preprocess])
        print(text_tokenize)
        prediction = model.predict_classes(text_tokenize)

        if int(prediction[0][0]) == 1:
            output = "positive"
        else:
            output = "negative"
        return render_template('index.html', prediction_text='This review sentiment is {}'.format(output))

    # text = [x for x in request.form.values()][0]
    # text_preprocess = preprocessing(text)
    # text_tokenize = tokenize([[text_preprocess]])
    # prediction = model.predict_classes(text_tokenize)
    
    # output = prediction[0]
    # return render_template('index.html', prediction_text='This review sentiment is {}'.format(output))
@app.route('/results',methods=['POST'])
def results():

    
    # print(text_tokenize)
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        data = request.get_json(force=True)
        text = list(data.values())[0]
        text_preprocess = preprocessing(text)
        text_tokenize = tokenize([text_preprocess])
        prediction = model.predict_classes(text_tokenize)
        # y_hat = keras_model_loaded.predict(predict_request, batch_size=1, verbose=1)
        # prediction = model.predict([np.array(list(data.values()))])
        # print(prediction)
        if int(prediction[0][0]) == 1:
            output = "positive"
        else:
            output = "negative"
        return jsonify(str(output))

if __name__ == "__main__":
    app.run(debug=True)