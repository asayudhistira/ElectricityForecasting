# Importing necessary libraries and modules.
import IPython # Importing IPython for interactive computing and creating interactive Python sessions.
import IPython.display # Importing display tools from IPython for displaying rich content such as images and videos.
import numpy as np # Importing numpy for numerical operations and array manipulations.
import matplotlib.pyplot as plt # Importing matplotlib for creating static, interactive, and animated visualizations in Python.
import tensorflow as tf # Importing TensorFlow, a machine learning and neural networks library.
import pandas as pd # Importing pandas for data manipulation and analysis, especially for handling tabular data.
import json # Importing the json module for parsing and generating JSON data.
import socket # Importing the socket library for networking support, such as creating server-client communication.

# Setting a constant for the maximum number of epochs (interations) during model training.
# Epoch = refers to the number of times the entire training dataset is passed forward and backward through the neural network. 
MAX_EPOCHS = 20

# Defining a function to compile and fit a TensorFlow model with "early stopping".
# Early stopping = Regularization method to prevent overfitting. Early stopping helps mitigate overfitting by 
# monitoring model performance on validation set during training and stopping training process when model performance on validation data starts degrading. 

# 1. Parameters (similar to defining a "recipe" in R):
#   - model: TensorFlow model to be compiled and fitted.
#   - window: The data window used for training the model.
#   - patience: The number of epochs with no improvement after which training will be stopped.
def compile_and_fit(model, window, patience=2):
    # Setting up early stopping to halt training when the validation loss has stopped improving.
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')
    # Compiling the model with a Mean Squared Error loss function and the Adam optimizer.
    model.compile(loss=tf.losses.MeanSquaredError(), optimizer=tf.optimizers.Adam(), metrics=[tf.metrics.MeanSquaredError()])
    # Fitting the model on the training data with a specified number of epochs, and using the validation data.
    history = model.fit(window.train, epochs=MAX_EPOCHS, validation_data=window.val, callbacks=[early_stopping])
    # Returning the training history, which contains information about the training process.
    return history

# 2. Defining function to prepare and train a machine learning model.
def train():
    # Reading data from a JSON file and loading it into a dictionary.
    df = {}
    with open(r"C:\Users\anton\Downloads\total_data.json", "r") as file:
        df = json.load(file)
    
    # 2.1. Converting the loaded data into a Pandas dataframe.
    df = pd.DataFrame(df)
    
    # 2.2. Selecting/Subsetting specific columns from pandas dataframe (df).
    df = df.filter(['hour', 'day', 'temperature', 'windSpeed', 'windDirection', 'cloudCover', 'timeslot', 'executionPrice', 'executionMWh'])
    
    # 2.3. Creating a dictionary to map column names (features) to their indices in dataframe (df). 
    # Indices are used to access specific columns (features) in dataset. 
    column_indices = {name: i for i, name in enumerate(df.columns)}
    
    # Setting input and output sequence lengths (adjust according to our dataset)
    # time steps = refer to discrete units of time at which observations or measurements are recorded.
    input_steps = 24 # number of past time steps used as input to the model
    output_steps = 12 # number of future time steps for which the model generates predictions
    
    # 3. Splitting the data into training, validation, and test sets.
    n = len(df) # number of rows (exlcuding headers)
    train_df = df[0:int(n * 0.7)] # training set contains 70% of complete dataset. 
    val_df = df[int(n * 0.7):int(n * 0.9)] # validation set contains from 0,7 to 0,9 portion (= 20%) of complete dataset
    test_df = df[int(n * 0.9):] # test set contains the last 10% of complete dataset. 
    
    # 4. Determining the number of features / columns in the dataframe starting from index 0.
    num_features = df.shape[1] # Adjust according to number of predictors/columns in dataframe 
    
    # 5. Standardizing the data by subtracting mean and dividing by standard deviation.
    train_mean = train_df.mean() # calculate and store mean
    train_std = train_df.std() # calculate and store std dev

    # 5.1. Use stored metrics to standardize train, validation and test set: 
    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    # 6. Defining a class for creating data windows for training and testing.
    # "Window" in time-series: fixed period of historical data used for making predictions or analyzing patterns, 
    # with the size of the window determining how far back in time the data is considered.
    class WindowGenerator():
        def __init__(self, input_width, label_width, shift, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None): # "Window Generator" specific parameters
            self.train_df = train_df # Store training data
            self.val_df = val_df # Store validation data
            self.test_df = test_df # Store test data
            self.label_columns = label_columns # Store label columns 
            if label_columns is not None: # If statement to check if columns are specified
                self.label_columns_indices = {name: i for i, name in enumerate(label_columns)} # Create a map of label column names to their positions.
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)} # Create a map of all label column names to their positions.
            self.input_width = input_width # Set how many past time steps to use as input (adjusted per defined value in 2.3.)
            self.label_width = label_width # Set how many future time steps to use as input (adjusted per defined value in 2.3.)
            self.shift = shift # Set the time step difference between input and prediction.
            self.total_window_size = input_width + shift # Calculate the total size of the window.
            self.input_slice = slice(0, input_width) # Define the slice for input data.
            self.input_indices = np.arange(self.total_window_size)[self.input_slice] # Create a list of indices for input data.
            self.label_start = self.total_window_size - label_width # Calculate where labels start.
            self.labels_slice = slice(self.label_start, None) # Define the slice for label data.
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice] # Create a list of indices for label data.

    # 6.1. Function to split a window of data into inputs and labels.
    def split_window(self, features):
        inputs = features[:, self.input_slice, :]  # Extract input data from the features based on input_slice.
        labels = features[:, self.labels_slice, :]  # Extract label data from the features based on labels_slice.
        if self.label_columns is not None:
            # Stack selected label columns if specified.
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None])  # Set the shape of the inputs.
        labels.set_shape([None, self.label_width, None])  # Set the shape of the labels.
        return inputs, labels  # Return the processed inputs and labels.
    
    # 6.2. Function to plot the data window with options to overlay model predictions.
    def plot(self, model=None, train_mean=0, train_std=0, plot_col='executionPrice', max_subplots=5):
        inputs, labels = self.example  # Retrieve example input and label data.
        plt.figure(figsize=(12, 8))  # Set the size of the plot.
        plot_col_index = self.column_indices[plot_col]  # Get the index of the column to plot.
        max_n = min(max_subplots, len(inputs))  # Determine the number of subplots to create.
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)  # Create a subplot for each example.
            inputs_n = inputs[n, :, plot_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]  # Normalize the input data.
            plt.plot(self.input_indices, inputs_n, label='Inputs', marker='.', zorder=-10)  # Plot the inputs.

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)  # Get the index for the label column.
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue

            labels_n = labels[n, :, label_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]  # Normalize the label data.
            plt.scatter(self.label_indices, labels_n, edgecolors='k', label='Labels', c='#2ca02c', s=64)  # Plot the labels.
            if model is not None:
                predictions = model(inputs)  # Generate predictions using the model if provided.
                predictions = predictions[n, :, label_col_index] * train_std[plot_col_index] + train_mean[plot_col_index]
                plt.scatter(self.label_indices, predictions, marker='X', edgecolors='k', label='Predictions', c='#ff7f0e', s=64)  # Plot the predictions.

            if n == 0:
                plt.legend()  # Add a legend in the first subplot.

        plt.xlabel('Timeslots [h]')  # Set the x-axis label.

    # 6.3. Function to create a TensorFlow dataset from the provided data.
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)  # Convert data to a numpy array of type float32.
        ds = tf.keras.utils.timeseries_dataset_from_array(data=data, targets=None, sequence_length=self.total_window_size, sequence_stride=1, shuffle=True, batch_size=32)  # Create a timeseries dataset.
        ds = ds.map(self.split_window)  # Apply the split_window function to each element in the dataset.
        return ds  # Return the processed dataset.

    @property # Property to get the training dataset.
    # 6.4. Create and return the training dataset using the training data frame:
    def train(self):  
        return self.make_dataset(self.train_df)  

    @property # Property to get the validation dataset.
    # 6.5. Create and return the validation dataset using the validation data frame: 
    def val(self):
        return self.make_dataset(self.val_df)  

    @property # Property to get the test dataset.
    # 6.6. Create and return the test dataset using the test data frame: 
    def test(self):
        return self.make_dataset(self.test_df)  

    @property # 6.7. Property to get an example batch of data:
    def example(self):
        result = getattr(self, '_example', None)  # Try to retrieve a cached example.
        if result is None:
            result = next(iter(self.train))  # If no cached example, get a new one from the training dataset.
            self._example = result  # Cache the new example for future use.
        return result  # Return the example batch of data.
        
    # 6.8. Define the number of steps to predict in the future for multi-step forecasting.
    OUT_STEPS = 24
    # 6.8.1. Initialize a WindowGenerator object for multi-step time series forecasting.
    multi_window = WindowGenerator(input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS) # The WindowGenerator will use a window of 24 time steps to predict the next 24 steps.
    # 6.8.2. Initialize a dictionary to store performance metrics of the model on the validation dataset.
    multi_val_performance = {} # This will be used for evaluating the model's performance in multi-step forecasting.
    # 6.8.3. Initialize a dictionary to store performance metrics of the model on various datasets.
    multi_performance = {} # This dictionary is intended for tracking and comparing model performance across different datasets.

    # 6.9. Creating a sequential LSTM model for multi-step time series forecasting.
    multi_lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(OUT_STEPS*num_features,kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])

    # 6.10. Compile and fit the model using the defined window generator.
    history = compile_and_fit(multi_lstm_model, multi_window)
    
    # Clear the output of the IPython display to keep the output clean.
    IPython.display.clear_output()

    # 6.11. Evaluate model performance:
    # 6.11.1. Evaluate the model on the validation dataset and store the performance metrics.
    multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
    # 6.11.2. Evaluate the model on the test dataset and store the performance metrics.
    multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
    # 6.11.3. Plot the model predictions along with the actual data.
    multi_window.plot(multi_lstm_model,train_mean,train_std)
    # 6.11.4. # Return the trained model along with the mean and standard deviation of the training dataset, and the length of the dataset.
    return multi_lstm_model,train_mean,train_std,n

# 6.11.5. Call the train function and store the returned values.
multi_lstm_model,train_mean,train_std,n = train()

# 6.12. Define the predict function to generate predictions using the trained LSTM model.
def predict(multi_lstm_model,n):
    label_col_index = 7 # Index of the label column in the dataset.
    plot_col_index = 7 # Index of the column to plot.
    predictions = multi_lstm_model # Placeholder for actual prediction logic.
    predictions = predictions[n, :, label_col_index] * train_std[plot_col_index] + train_mean[plot_col_index] # Apply normalization using training mean and standard deviation.
    return predictions


# ------------------------------------------------------
# ------------------------------------------------------
# 6.13. Server setup for network communication.
PORT = 8098  # Define the port number for the server.
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # Create a TCP/IP socket.
print(sock)  # Print the socket object for debugging purposes.
server_address = ('localhost', PORT)  # Set the server address ('localhost' indicates the same machine).
sock.bind(server_address)  # Bind the socket to the address.
sock.listen(1)  # Listen for incoming connections (number indicates the backlog of connections allowed).
c = 0  # Initialize a counter.

# 6.14. Begin an infinite loop to handle client requests.
while True:
    print('\nwaiting for a connection\n')
    # Wait for a client to connect.
    connection, client_address = sock.accept()

    # Receive data from the client.
    data = connection.recv(16)
    data = data.decode('ascii')  # Decode the data from ASCII.
    print(data)  # Print the received data for debugging.
    whole_data = data.rstrip()  # Strip any trailing whitespace from the data.
    data = data.rstrip().split()[0]  # Extract the first word from the data for command identification.

    # 6.15. Handle 'train' command from client.
    if data == "train":
        out = "\nok\n"  # Prepare an acknowledgment message.
        connection.sendall(out.encode('utf-8'))  # Send the acknowledgment back to the client.
        train()  # Call the train function to train the model.

    # 6.16. Handle 'predict' command from client.
    elif data == "predict":
        out = predict()  # Call the predict function and store its output.
        connection.sendall(out.encode('utf-8'))  # Send the prediction result back to the client.

    # 6.17. Handle any other unrecognized commands.
    else:
        print("error")  # Print an error message for any unrecognized commands.

    c = c + 1  # Increment the counter for each iteration/connection.
# ------------------------------------------------------
# ------------------------------------------------------
