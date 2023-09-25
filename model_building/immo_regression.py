import numpy as np
import pandas as pd
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

def cleaning(df):
    '''This function returns a dataset with the following modifications:
    1. Features renamed - such as 'swimming-pool' transformed to 'swimming_pool' in order to avoid computing problems in python;
    2. Removal of duplicated rows;
    3. NaN replaced with 0;
    4. Removal of unnecessary for the price features based on the insights acquired on the data analysis part.'''
    print('=='*30) 
    # Renaming some columns names in order to avoid problems
    df.rename(columns={'swimming-pool': 'swimming_pool'}, inplace=True)
    df.rename(columns={'state-building': 'state_building'}, inplace=True)
    df.rename(columns={'land-surface': 'land_surface'}, inplace=True)
    # Working with 0 values on the column "state_building"
    df.drop(df[df['state_building'] == "0"].index, inplace= True)
    df['state_building'] = df['state_building'].astype(str)
    # Removing duplicated rows
    df.duplicated()
    # Replacing NaN with 0
    df.fillna(0, inplace=True)   
    # Removing all the rows where there's no information on locality
    df.drop(df[df['locality'] == 0].index, inplace=True)
    # Removing the columns type-transaction and url as it is not relevant
    df.drop(columns=['type-transaction'], inplace=True)
    df.drop(columns=['url'], inplace=True)
    # Removing the columns area garden and terrace
    df.drop(columns=['area_terrace'], inplace=True)
    df.drop(columns=['area-garden'], inplace=True)
    # Removing the column number of facades
    df.drop(columns=['n-facades'], inplace=True)
    print('Features renamed and dropped after the cleaning function.')
    print('=='*30)    
    return df

def model (df):
    '''This function takes the preprocessed dataset, builds and trains a model using Random Forest Regressor and returns:
    1. The accuracy of the model for the trainning and test set;
    2. The RMSE of the model.  
    The selection of the model was made after an investigation of the different linear and
    non-linear types of models regarding the best score acquired considering the relationship between 
    the price and the variables in the present dataset.
    If you want to check it up this investigation, go to the folder model_building and run the jupyter notebooks presented on it.
    '''
    # Dropping null values
    df.dropna(inplace=True) 
    # # Assigning X and y
    X = df.drop(['price'], axis=1)
    y = df['price']
    # Pipeline building
    # During the first tranformation, the categorical values will be transformed into numeric values with OneHotEncoder
    trans_1 = ColumnTransformer([('ohe_trans', 
                                  OneHotEncoder(sparse_output=False, 
                                                handle_unknown='ignore'), 
                                  [0, 1, 2, 12])], 
                                remainder='passthrough' )
    # The second transformation normalizes the values
    trans_2 = ColumnTransformer([('scale', MinMaxScaler(),slice(0,len(X)+1))], remainder='passthrough')
    # The third transformation receives the model itself
    trans_3 = RandomForestRegressor(random_state=3)
    # Instanting the pipeline with all the transformations happening in steps
    model = Pipeline(steps=[('trans_1', trans_1),
                            ('trans_2', trans_2),
                            ('trans_3', trans_3)])
    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Trainning the model
    model.fit(X_train, y_train)
    # Informing to the user the type of model used
    print('The model selected was Random Forest Regressor!')
    print('It was set everything by default, despite the random state (equal to 3)')
    print('=='*30) 
    print('The accuracy of this model for the dataset can be checked bellow: ')
    # Checking up the model score for the trainning set
    print('Training set score: ', model.score(X_train, y_train))
    # Checking up the model score for the test set
    print('Test set score: ', model.score(X_test, y_test))
    # Calculating the RMSE
    pred = model.predict(X_test)
    rmse = np.sqrt(MSE(y_test, pred))
    shape_model = X_train.shape
    print('the shape of X_train is: ', shape_model)
    print("RMSE: % f" %(rmse))
    print('=='*30)
    # Saving the model with pickle
    filename = "immo_prediction_model.pickle "
    pickle.dump(model, open(filename, "wb"))
    # Loading the model to make predictions
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model
        
        