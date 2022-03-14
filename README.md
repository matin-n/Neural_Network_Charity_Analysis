# Neural Network Charity Analysis

## Project Overview
This project utilizes machine learning and neural networks to create a binary classifier capable of predicting whether applications will be successful if funded by Alphabet Soup. The goal of this project is to achieve 75% accuracy.

The dataset contains more than 34,000 organizations that have received funding. Within the dataset, many columns capture the metadata about each organization, structured as the following:

|                            | **Captured Metadata**                  | **Data Type** |
|----------------------------|----------------------------------------|---------------|
| **EIN**                    | Identification column                  | int64         |
| **NAME**                   | Identification column                  | object        |
| **APPLICATION_TYPE**       | Alphabet Soup application type         | object        |
| **AFFILIATION**            | Affiliated sector of industry          | object        |
| **CLASSIFICATION**         | Government organization classification | object        |
| **USE_CASE**               | Use case for funding                   | object        |
| **ORGANIZATION**           | Organization type                      | object        |
| **STATUS**                 | Active status                          | int64         |
| **INCOME_AMT**             | Income classification                  | object        |
| **SPECIAL_CONSIDERATIONS** | Special consideration for application  | object        |
| **ASK_AMT**                | Funding amount requested               | int64         |
| **IS_SUCCESSFUL**          | Was the money used effectively         | int64         |


## Results
### Data Processing

- Target variable: `IS_SUCCESSFUL`
- Features: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
- Non-beneficial: `EIN`, `NAME`

#### Processing Steps
1. Drop non-beneficial columns
2. Determine the number of unique values in each column
3. Bin `APPLICATION_TYPE` & `CLASSIFICATION`
4. Encode categorical variables with [OneHotEncoder()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
5. Split pre-processed data into features (`X`) and target (`y`)
6. Split pre-processed data into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) dataset
8. Scale the data with [StandardScalar()](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

### Compile, Train and Evaluate the Model

#### Model Summary
The summary of the first model is as structured:

Model: "sequential"

| **Layer (type)** | **Function** | **Output Shape** | **Param #** |
|------------------|--------------|------------------|-------------|
| dense (Dense)    | relu         | (None, 132)      | 5940        |
| dense_1 (Dense)  | relu         | (None, 88)       | 11704       |
| dense_2 (Dense)  | sigmoid      | (None, 1)        | 89          |

```python
# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_inputs = len(X_train[0])
hidden_nodes_layer1 = len(X_train[0]) * 3
hidden_nodes_layer2 = len(X_train[0]) * 2

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(
    tf.keras.layers.Dense(
        units=hidden_nodes_layer1, activation="relu", input_dim=number_inputs
    )
)

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Check the structure of the model
nn.summary()
```

For selecting the number of units per layer, I followed the principle::
> "A good rule of thumb for a basic neural network is to have two to three times the amount of neurons in the hidden layer as the number of inputs."

The input layer and hidden layer are set to the [ReLU function](https://keras.io/api/layers/activations/#relu-function). The ReLU function was chosen because of its known simple computation (relative to other functions), which reduces the training and evaluation time. Additionally, the convergence time with ReLU is quick. In short, training with ReLU gives better performance and faster convergence.

The output layer is set to the [sigmoid function](https://keras.io/api/layers/activations/#sigmoid-function) since the target variable is a binary classification. The sigmoid function returns a value between 0 and 1.

#### Model Compilation
```python
# Create a callback that saves the model's weights every epoch
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, verbose=0, save_weights_only=True, period=5
)

# Compile the model
nn.compile(
    optimizer=keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"]
)
```

- Optimizer: [Adam()](https://keras.io/api/optimizers/adam/)
- Probabilistic losses: [BinaryCrossentropy](https://keras.io/api/losses/probabilistic_losses/#binarycrossentropy-class)
- Metrics: [Accuracy](https://keras.io/api/metrics/accuracy_metrics/)


#### Training

```python
# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=50, callbacks=[cp_callback])
```
- Fifty epochs are set, and a call back was created to save the model every five epoch(s).

#### Results
```python
# Evaluate the model using the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
- Accuracy: 0.7286
- Loss: 0.5567

## Summary

Reaching the goal of 75% accuracy was not met. However, further adjustments may be able to increase the accuracy. For instance, the following adjustments may have potential improvement:
1. [Activation function(s)](https://keras.io/api/layers/activations/#available-activations) could be modified
2. [Dropout() function](https://keras.io/api/layers/regularization_layers/dropout/) applied to reduce overfitting
3. [BatchNormalization()](https://keras.io/api/layers/normalization_layers/batch_normalization/) applied to normalize inputs
4. [HyperParameter(s)](https://keras.io/api/keras_tuner/hyperparameters/) with KerasTuner to find best values
5. Optimization & fine-tuning of preprocessing

Additionally, [transfer learning](https://www.tensorflow.org/guide/keras/transfer_learning) may be applied to a pre-trained model to train new predictions on a dataset. Luckily, there are many potential paths to improve a model, many of which were not stated!
