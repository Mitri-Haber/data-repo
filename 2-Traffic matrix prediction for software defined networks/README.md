# Traffic matrix prediction for software defined networks

## 1- Introduction

Traffic Matrix clearly describes the volume and the distribution of traffic flows inside a network, the rows and columns are nodes, could be a central node or an edge node.

Each element of the traffic matrix is the outgoing traffic in MB from the row element to the column element at a certain time step, For exp:

The outgoing traffic from Node LAX to NODE NYC is 2389 MBs for the slice in time we have below.

![image](https://user-images.githubusercontent.com/109002028/197401372-5f083eb3-53d4-401f-878d-9425ee92d3b1.png)

Traffic matrices can play a vital role in improving network management, such as traffic accounting, short-time traffic scheduling or re-routing, network design, and most importantly resource planning and saving cost on unwanted infrastructure. 

Predicting the next steps of the matrix accurately is essential to handle these tasks.

## 2- Goal

A discrete traffic tensor can be represented as a 3D tensor where each 2D slice is a traffic matrix at specific time step. A time step separates two discrete traffic matrices in time, the time window is actually a parameter that needs to be optimized and is compromised between model feasibility and the business needs.

![image](https://user-images.githubusercontent.com/109002028/197450753-39e00655-d596-477b-8877-181431e37769.png)



The goal of this work is to have a neural network that takes as input 10 traffic matrices and outputs the prediction of the 11th one.

![image](https://user-images.githubusercontent.com/109002028/197448059-e7b1bf54-bb18-4d4a-8a0f-e447f1e17ddf.png)




To do so, the below CONV-LTSM model has been built and trained on static data that belongs to Abilene network.

![model](https://user-images.githubusercontent.com/109002028/197463864-8b14bc73-c402-41c8-87a1-c98ccaa70cd0.png)


In the process of training the model, hyperband tuning has been used to find the optimal layer sizes, this was done in the script "cnn-ltsm-10-hyperband.py".
## 3- Loss and errors.

The loss and metric for this model should be MAE, because when using MSE we will square errors, and the normalized values are between 0-1, squaring an error smaller than 1 will give a smaller number, hence a false sense of optimization.

In previous trainings for this model HUBER loss was used, this will return MAE when the error is between 0-1, and MSE if the error is larger than 1.

Therefore, we will penalize large errors using MSE, and will not have a false sense of optimization when errors are smaller than 1.

## 4- Training, validation

After multiple trainings and modifications, the final training was done in "train-cnn-ltsm-10.py"

The validation of this model is not straight forward, unlike other models, this model is a time series model with multiple features.

For example, each input and output is a 12 x 12 matrix  (144 feature), where we need to have precision on each feature.

Since we have a lot of features, the aggregation of MAE into a single number will dissolve the meaning and contribution of the Error because of the high dimensionality.

For exp: the prediction of a 20 X 20 traffic matrix (400 feature), could have 399 of the values predicted precisely except for 1 and this could be catastrophic when using the traffic matrix to operate networks. 

Because we are dividing the error of this feature by a high dimension, we will lose its importance.

Other than the aggregated singular MAE, we need an MAE for each feature.

The below visualizes a 12 * 12 MAE matrix along the time steps of the validation data. 

![image](https://user-images.githubusercontent.com/109002028/197463685-49a74814-fd01-4e78-9fbd-8a080ffa5104.png)

The below visualizes a 12 * 12 standard deviation MAE matrix along the time steps of the validation data.

![image](https://user-images.githubusercontent.com/109002028/197476485-492d1cc8-27a3-4429-aaa7-d02b9c57bb1c.png)

The above are better metrics that will provide us with better insights for reaching a precise model.

## 5- Errors, production, and data drift.

In this work, we got reasonable errors for a POC, however this model cannot be put in production, we need much more data to train on.

The data points are not enough to train such complex model, with more data from the internet provider, it's possible to reach a much more precise model.

It's challenging to reach a perfect precision with such models, however in production error corrections could be added to the matrix, for example for each feature we could continuously compute the MAE and std of MAE of previous steps  and add the error for each feature.

To handle data drift, we need two prediction/training engines, one engine will be training the model on new data, and the other is serving predictions.

A model controller app is needed, the app has multiple functionalities, it will receive prediction data from the client and sends it to the elected prediction engine.

It will continuously monitor the prediction feedback from the elected prediction engine, while letting the elected training engine know that it should keep training the model.

Once the controller discover an MAE higher than a certain threshold it will switch the functionalities of the engines.

Upon each switch the engines will share the newley trained model.


![image](https://user-images.githubusercontent.com/109002028/197483540-6843adaa-b130-4aa7-b9d5-300440f5e764.png)





