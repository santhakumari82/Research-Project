# Filename: models.py
# Description: This python program is used to build the machine learning and deep learning models to predict crypto price

# import python libraries
import os
import pandas as pd
import numpy as np
import math
import datetime as dt
from datetime import datetime, timezone

# For Evalution we will use these library
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score

# For preprocessing we will use these library
from sklearn.preprocessing import MinMaxScaler

# For fbprophet model building we will use these library
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

# For LSTM model building we will use these library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM


# For Plotting we will use these library
import matplotlib.pyplot as plt

class Models:
    
    def buildFBProphetModel(self,data,symbol,symbol_desc):        
        print("Building FBProphet Model")
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        data = data[(data['Date'] >= "2018-01-01") & (data['Date'] <= "2022-11-20")]
        print("printing data shape..")
        print(data.shape)
        # We will be keeping only the date and closing price columns
        df_model = data.drop({'Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock_Splits', "Crypto_Symbol", "Creation_Time"}, axis=1)
        print(df_model.head(5))
        title_color = '#884EA0'
        # Renaming the date column to 'ds' and close price to 'y'
        df_model.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    
        # Training and Testing of the dataframe
        # We will take first 22 months of the data and training it, last month will be tested on
        df_model['ds'] = pd.to_datetime(df_model['ds'], format='%Y-%m-%d')
        
        split = int(0.8*len(df_model))
        train_df = df_model[0:split]
        test_df = df_model[split:]
        print('Predicting yhat and its upper and lower bound values')
        # Print the number of records and date range for training and testing dataset.
        print('The training dataset has', len(train_df), 'records, ranging from', train_df['ds'].min(), 'to', train_df['ds'].max())
        print('The testing dataset has', len(test_df), 'records, ranging from', test_df['ds'].min(), 'to', test_df['ds'].max())
                
        # Training the Time Series Model Using FBProphet
        # Prophet model for time series forecast
        # Creating the FBProphet model with confidence internal of 95%
        fb_model = Prophet(interval_width=0.95, n_changepoints=7)

        # Fitting the model using the training dataset
        fb_model.fit(train_df)

        # Creating a future dataframe for prediction
        future_df = fb_model.make_future_dataframe(periods=31)
        # Forecasting the future dataframe values
        forecast_df = fb_model.predict(future_df)
        # Checking the forecasted values and upper/lower bound
        print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Visualizing the model 
        fig = fb_model.plot(forecast_df)
        #plt.title('Predicted Price of Bitcoin')
        ax = fig.gca()
        ax.plot( test_df["ds"], test_df["y"], 'r.')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price (CAD)')
        ax.set_title('Training and Testing data split of '+symbol_desc,fontweight="bold",size=12,color=title_color,pad='1.8')            
        fig.show()

        # Visualizing the components 
        fig = fb_model.plot_components(forecast_df)
        ax = fig.gca()
        plt.figure(figsize=(10, 5), tight_layout=True)
        fig.show()

        fig = fb_model.plot(forecast_df)
        ax = fig.gca()
        change_points = add_changepoints_to_plot(fig.gca(), fb_model, forecast_df)
        ax.set_title('Change points of '+symbol_desc+" between 2018-2022",fontweight="bold",size=12,color=title_color,pad='1.8')
        #plt.figure(figsize=(10, 5), tight_layout=True)
        fig.show()
        

        # Listing the change points in the dataframe
        print(f'The number of change points in this dataframe are {len(fb_model.changepoints)} . \nThe change points dates are \n{df_model.loc[df_model["ds"].isin(fb_model.changepoints)]}')
        print(" A comparison between the actual and predicted values with the cross validation in this time series model.")
        print("Mean Absolute Error of this model, which sums the difference between the actual and predicted value, divided by the number of predictions")
        # Cross validation
        cv_df = cross_validation(fb_model, initial='900 days', period='100 days', horizon = '100 days', parallel="processes")
        print("Cross validation of FBprophet...")
        """print("printing top records...")
        print(cv_df.head())"""
        print("printing tail records...")
        print(cv_df.tail())
       
        
        # Prophet Model Performance Evaluation
        # Model performance metrics
        prophet_df = performance_metrics(cv_df)
        print("Performance metrics of "+symbol_desc)
        from tabulate import tabulate
        pdtabulate=lambda prophet_df:tabulate(prophet_df,headers='keys',tablefmt='psql')
        print(pdtabulate(prophet_df))
        
        # Visualize the performance metrics based on Mean absolute error metric
        fig = plot_cross_validation_metric(cv_df, metric='mae', rolling_window=0.1)
        ax = fig.gca()
        ax.set_title(symbol_desc+' - Performance metrics of prediction based on MAE',fontweight="bold",size=12,color=title_color)
        ax.set_xlabel('Prediction days')
        ax.set_ylabel('mae')       
        fig.show()
        

    def buildLSTMModel(self,df_crypto,symbol,symbol_desc):
        print("Building LSTM Model...")
        df_crypto['Date'] = pd.to_datetime(df_crypto['Date'], format='%Y-%m-%d')
        title_color = '#884EA0'
        # Filtering the dataframe to have data from 2020 to 2022
        df_crypto = df_crypto[(df_crypto['Date'] >= "2018-01-01") & (df_crypto['Date'] <= "2022-11-20")]

        # Selecting only Bitcoin cryptocurrency
        df_model = df_crypto[(df_crypto['Crypto_Symbol'] == symbol)]
        print(df_model.shape)
        print(df_model.head(5))

        # We will be keeping only the date and closing price columns
        
        model_data = df_model['Close']
    
        # Normalizing the dataset using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0,1))
        model_data = scaler.fit_transform(np.array(model_data).reshape(-1,1))
        print("Model shape:",model_data.shape)

        print("Data has null values?",np.isnan(model_data).any())

        #we are going to take a seq len of 100 days and predict 100 days in future 
        SEQUENCE_LEN = 100

        def make_sequences(pre_data, seq_len):
            ds = []
            for index in range(len(pre_data) - SEQUENCE_LEN):
                ds.append(pre_data[index: index + SEQUENCE_LEN])
            return np.array(ds)

        def prepare_data(pre_data, seq_len, train_split):
            data = make_sequences(pre_data, SEQUENCE_LEN)
            num_train = int(train_split * data.shape[0])
            #Splitting into X and Y training data
            X_train = data[:num_train, :-1, :]
            Y_train = data[:num_train, -1, :]
            #Splitting into X and Y test data
            X_test = data[num_train:, :-1, :]
            Y_test = data[num_train:, -1, :]

            return X_train, X_test, Y_train, Y_test

        X_train, X_test, Y_train, Y_test = prepare_data(model_data, SEQUENCE_LEN, train_split = 0.80)

        print("X_train:", X_train.shape)

        print("X_test:", X_test.shape)

        # Preparing for the model
        DROPOUT = 0.2
        WINDOW_SIZE = SEQUENCE_LEN - 1

        tf_model=tf.keras.Sequential([
            tf.keras.layers.LSTM(128,input_shape=(WINDOW_SIZE, X_train.shape[-1]),return_sequences=True),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.LSTM(128,return_sequences=True),
            tf.keras.layers.LeakyReLU(alpha=0.5),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.LSTM(64,return_sequences=False),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1,activation='linear')
        ]
        )
        print("LSTM Model Summary")
        print(tf_model.summary()) # _squared_
        tf_model.compile(loss='mean_absolute_error',optimizer='adam')

        BATCH_SIZE = 64

        tf_model_history = tf_model.fit(X_train,Y_train, epochs=100, batch_size = BATCH_SIZE, validation_split=0.1, shuffle=False)

        plt.plot(tf_model_history.history['loss'])
        plt.plot(tf_model_history.history['val_loss'])
        plt.legend(['train','test'], loc='upper left')
        plt.title('Model loss',fontweight="bold",size=16,color=title_color)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        # prediction of lstm model
        train_predict = tf_model.predict(X_train)
        
        y_hat = tf_model.predict(X_test)
        print("Train predict shape:", train_predict.shape)
        print("Y hat shape:", y_hat.shape)
        
        y_train_inverse = scaler.inverse_transform(train_predict)        
                
        y_hat_inverse = scaler.inverse_transform(y_hat)

        original_ytrain = scaler.inverse_transform(Y_train.reshape(-1,1))
        original_ytest = scaler.inverse_transform(Y_test.reshape(-1,1))

        print("Train data MAE: ", mean_absolute_error(original_ytrain,y_train_inverse))
        print("Test data MAE: ", mean_absolute_error(original_ytest,y_hat_inverse))

        print("Train data MAPE: ", mean_absolute_percentage_error(original_ytrain,y_train_inverse))
        print("Test data MAPE: ", mean_absolute_percentage_error(original_ytest,y_hat_inverse))
        #y_test_inverse
        plt.plot(original_ytest, label="Actual Price", color='green')        
        plt.plot(y_hat_inverse, label="Predicted Price", color='red')
        
        plt.title(symbol_desc+" Price Prediction",fontweight="bold",size=16,color=title_color,pad='2.0')
        plt.ylabel("Close Price(CAD)")
        plt.xlabel("Time [days]")
        plt.legend(loc='best')
        
        plt.show()


 
                 

        

        
