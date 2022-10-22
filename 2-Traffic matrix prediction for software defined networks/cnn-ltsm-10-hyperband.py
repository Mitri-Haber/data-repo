import pandas as pd
import numpy as np
from math import ceil
import tensorflow as tf
from tensorflow.keras.layers import Input,ConvLSTM2D,BatchNormalization,Conv2D,Add
import keras_tuner as kt


file_name = "./Abilene-OD_pair.txt"
df = pd.read_csv(file_name)
data_columns = list(df.columns.values)
data_columns.remove('time')
data = df[data_columns].values
data = np.clip(data, 0.0, np.percentile(data.flatten(), 99))  # we use 99% as the threshold
df[data_columns] = data
timesteps=data.shape[0]
df['time']=df.index
times= df['time']
max_list = np.max(data, axis=0)
min_list = np.min(data, axis=0)
data = (data - min_list) / (max_list - min_list)
data[np.isnan(data)] = 0
data[np.isinf(data)] = 0
x_data = []
for i in range(timesteps ):
    x=data[i]
    x_data.append(x)

x_data = np.array(x_data)

x_data_2d=x_data.reshape((48096,12,12))
x_data_2d.shape

split_time=int(timesteps*0.8)
time_train = times[:split_time]
x_train = x_data_2d[:split_time]
time_valid = times[split_time:]
x_valid = x_data_2d[split_time:]
feature_count=[12,12]
window_size = 10
batch_size = 32
shuffle_buffer_size = 1000
conv_input=[window_size,12,12]

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):

    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    
    return dataset

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
history = model1.fit(train_set, epochs=10) #callbacks=[lr_schedule]

tf.keras.backend.clear_session()

def model_builder(hp):
    input=Input(shape=(None,12,12,1))

    hp_filters1 = hp.Int('units1', min_value=32, max_value=256, step=32)

    h=ConvLSTM2D(filters=hp_filters1, kernel_size=(3,3),padding='same',activation="relu", return_sequences=True, name="1rst-CONV-LTSM", data_format='channels_last')(input)
    h=BatchNormalization()(h)

    hp_filters2 = hp.Int('units2', min_value=32, max_value=256, step=32)

    h=ConvLSTM2D(filters=hp_filters2, kernel_size=(3,3),padding='same', return_sequences=True,name="2nd-CONV-LTSM")(h)
    h=BatchNormalization()(h)

    hp_filters3 = hp.Int('units3', min_value=16, max_value=128, step=16)

    h=ConvLSTM2D(filters=hp_filters3, kernel_size=(3,3) , padding='same', return_sequences=True,name="3rd-CONV-LTSM")(h)
    h=BatchNormalization()(h)

    hp_filters4 = hp.Int('units4', min_value=16, max_value=128, step=8)

    h=ConvLSTM2D(filters=hp_filters4, kernel_size=(3,3) , padding='same', return_sequences=False,name="4rt-CONV-LTSM")(h)
    h=BatchNormalization()(h)


    k=Conv2D(filters=1, kernel_size=(3,3),activation='sigmoid',padding='same', data_format='channels_last')(h)
    l=Conv2D(filters=1, kernel_size=(3,3),activation='sigmoid',padding='same', data_format='channels_last')(h)
    
    output=Add()([k,l])

    model=tf.keras.models.Model(inputs=input, outputs=output)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07,)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer,metrics=["mse"])

    return model


tuner = kt.Hyperband(model_builder,
                 objective='loss',
                 max_epochs=4,
                 factor=3,
                 directory='./Hyperband-Tuning',
                 project_name='kt_hyperband')

tuner.search(train_set, epochs=4)

best_hps=tuner.get_best_hyperparameters()[0]

print(f"""
The hyperparameter search is complete.
The optimal number of units is as below:
layer1 {best_hps.get('units1')} 
layer2 {best_hps.get('units2')} 
layer3 {best_hps.get('units3')} 
layer4 {best_hps.get('units4')} 
""")
