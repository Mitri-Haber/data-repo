from flask import Flask, request
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('cnn-ltsm10.h5')

app = Flask(__name__)
def model_forecast1(model, series, window_size=10, batch_size=1):

    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    forecast = model.predict(dataset)
    
    return forecast

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    data = np.array(data['data'])
    prediction = model_forecast1(model, data, window_size=10, batch_size=1)
    return {"result": prediction.tolist()}

if __name__ == "__main__":
    app.run(debug=True)

