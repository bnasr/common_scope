'''
Author: Bijan Seyednasrollah
Date: March 31, 2021

Python module to load, preprpcess, and analyze common scope estimates based on home features.

The purpose of this module is to facilitate data modeling.

'''

import pandas as pd

import logging

import re

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import Normalizer, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor    


def summarize(df):
    '''
    Takes in a dataframe, returns a summary table of the following:
        - column_name
        - data_type
        - unique_values
        - percent_missing
        
    Args:
        df (pandas dataframe): input dataframe
        
    Returns:
        pandas dataframe: The return value
    '''

    columns_summary = pd.DataFrame({'column_name': df.columns,
                                    'data_type': df.dtypes,
                                    'unique_values': df.nunique(),
                                    'percent_missing': df.isnull().sum() * 100 / len(df)})

    columns_summary = columns_summary.sort_values( by = ['data_type', 'percent_missing'])

    return columns_summary


def drop_colinearity(df, 
                     verbose = False,
                     thresh=10.0):
    
    '''
    Takes in a dataframe, drops colinear columns and retruns the rest.
        
    Args:
        df (pandas dataframe): input dataframe
        verbose (boolean): whether print out the progress
        threshold (float): maximum VIF score
    Returns:
        pandas dataframe: The return dataframe
    '''
    
    variables = list(range(df.shape[1]))
    dropped = True
    
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df.iloc[:, variables].values, ix)
               for ix in range(df.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            if verbose:
                print('dropping \'' + df.iloc[:, variables].columns[maxloc] +
                      '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    
    return variables



class CommonScope:
    '''
    Common scope data are loaded, trained and predicted in one place.
    
    Args:
        train_file (string): file path to the training dataset
        holdout_file (string): file path to the holdout dataset
        drop_colinears (bool): whether to drop colinear variables based on VIF.
        model (string): "LinearRegression", "Lasso", "Ridge" or a model function.

    
    Attributes:
        train_file (string): file path to the training dataset
        holdout_file (string): file path to the holdout dataset        
        columns (dict): dictionary of column names with their type
        train_data (dataframe): training dataset
        holdout_data (dataframe): holdout dataset
        imputers (dict): imputers for the data types
        preprocessed_feature (dataframe): preprocessed training features data
        drop_colinears (bool): whether to drop colinear variables based on VIF.
        model (string): "LinearRegression", "Lasso", "Ridge" or a model function.
    '''
    
    def __init__(self, 
                 train_file = None,
                 holdout_file = None,
                 drop_colinears = False,
                 model = 'LinearRegression'
                ):
        
        self.train_file = train_file
        self.holdout_file = holdout_file
        
        self.columns = self.__group_columns()
        
        self.train_data = None
        self.holdout_data = None
        self.imputers = None
        self.preprocessed_feature = None
        self.included_variables = None
        self.drop_colinears = drop_colinears
        self.model = model
        
        if train_file is not None:
            self.load_data(file_path = train_file)
        
        if holdout_file is not None:
            self.load_data(file_path = holdout_file, holdout = True)
        

    def __extract_feature_matrix(self, input_series):
        '''
        Internal method for parsing feature of bathrooms, bedrooms, etc.
        
        Args:
            input_series (Series): one-dimensional data of features
            
        Returns:
            processed data as binart matrix
        '''

        long_string = input_series.str.cat(sep=' ')

        features = set(re.sub(pattern = "\[|\]|'|,|_none_|", 
                   repl = '', 
                   string = long_string).split(sep = ' '))

        features.remove('')

        mat = pd.DataFrame(columns = features)

        for feature in mat.columns:
            mat[feature] = input_series.str.contains(feature) * 1

        return mat


    def load_data(self, file_path, holdout = False):
        '''
        Loads data from data file.
        
        Args:
            file_path (string): system path to the data file 
            holdout (bool): True or False whether the data set is holdout

        Returns:
            No return value. The data is loaded in the class.
        '''
        try: 
            df = pd.read_csv(file_path)
        except IOError as err:
            print(err)
            return 
        
        if holdout:
            self.holdout_data = df
            print(f'"{file_path}" was loaded as the hold-out data set: {df.shape[0]} x {df.shape[1]}')
        else:
            self.train_data = df
            print(f'"{file_path}" was loaded as the training data set: {df.shape[0]} x {df.shape[1]}')


    def __group_columns(self):
        '''
        Internal method to classify column names based on their types.
        
        Args: 
            No arguments needed.
        
        Return:
            dict: Data type to lists of column names
        '''
        
        bool_cols = ['has_renovation', 'in_gated_community']

        int_cols = ['bathrooms_full', 'bathrooms_half', 'bedrooms', 'exterior_stories',
                    'garage_spaces', 'pool', 'renovation_amount']

        float_cols = ['total_finished_sq_ft', 'basement_finished_sq_ft', 'sq_ft',
                      'above_grade_sq_ft', 'age', 'f_days_since_prev_close']

        object_cols = ['back_yard_condition', 'bathroom_condition', 'front_yard_condition',
                       'kitchen_appliance_type', 'kitchen_condition', 'kitchen_countertop',
                       'paint_condition', 'primary_floor_condition', 'primary_floor_type', 
                       'secondary_floor_condition', 'secondary_floor_type', 'market_name', 
                       'hvac_age', 'roof_age']

        features_cols = ['bathroom_features', 'eligibility_features', 'kitchen_features']

        date_cols = ['valuation_date']

        excluded_cols = ['flip_token', 'basement_unfinished_sq_ft', 'floor_condition', 
                         'floor_type', 'bathrooms', 'pool_above_ground']
        
        label_col = ['common_scope']
        
        return {'bool': bool_cols,
                'int': int_cols,
                'float': float_cols,
                'object': object_cols,
                'feature': features_cols,
                'date': date_cols,
                'exclude': excluded_cols,
                'label': label_col}

    
    def __set_imputers(self):
        '''
        Internal method to set imputers.
        
        Args:
            No input arguments needed.
        
        Returns:
            dict: Three imputers for integer, float and object columns.
        '''
        
        if self.train_data is None:
            raise('Training data is not loaded yet!')

        df = self.train_data
        
        int_cols = self.columns['int']
        float_cols = self.columns['float']
        object_cols = self.columns['object']

        self.imputers = {'int': SimpleImputer(strategy= 'median').fit(df[int_cols]), 
                         'float': SimpleImputer(strategy= 'mean').fit(df[float_cols]),
                         'object': SimpleImputer( strategy= 'most_frequent').fit(df[object_cols])}

        
    def __preprocessing(self, 
                       df):
        '''
        Internal method to preprocess the data.
        
        Args:
            df (dataframe): input dataframe to get processed.
            
        Retruns:
            dataframe: output processed dataframe.
        '''
        
        if self.train_data is None:
            print('Training data is not loaded yet!')
            return None
        
        int_cols = self.columns['int']
        float_cols = self.columns['float']
        object_cols = self.columns['object']
        date_cols = self.columns['date']
        features_cols = self.columns['feature']
        bool_cols = self.columns['bool']
        
        self.__set_imputers()

        df_imputed_int = pd.DataFrame(self.imputers['int'].transform(df[int_cols]),
                                      columns = df[int_cols].columns)
        
        df_imputed_float = pd.DataFrame(self.imputers['float'].transform(df[float_cols]),
                                                 columns = df[float_cols].columns)

        df_imputed_object = pd.DataFrame(self.imputers['object'].transform(df[object_cols]),
                                                 columns = df[object_cols].columns)
        
        df_imputed_object_dummies = pd.get_dummies(data = df_imputed_object, drop_first=True)

        df_date = pd.concat([pd.to_datetime(df[date_cols].iloc[:,0]).dt.year, 
                   pd.to_datetime(df[date_cols].iloc[:,0]).dt.month, 
                   pd.to_datetime(df[date_cols].iloc[:,0]).dt.day, 
                   pd.to_datetime(df[date_cols].iloc[:,0]).dt.dayofweek], 
                            axis = 1)
        
        df_date.columns = ['valuation_year', 'valuation_month', 'valuation_day', 'valuation_dayofweek']

        df_features_mat = pd.concat([self.__extract_feature_matrix(df[features_cols[0]]),
                                    self.__extract_feature_matrix(df[features_cols[1]]),
                                    self.__extract_feature_matrix(df[features_cols[2]])],
                                    axis = 1)

        df_wrangled = pd.concat([df_date, 
                                df[bool_cols]*1,
                                df_imputed_int,
                                #df_features_mat,
                                df_imputed_float,
                                df_imputed_object_dummies
                                ], axis = 1)

        df_wrangled = pd.concat([df_wrangled], axis = 1)
        
        return df_wrangled


    def fit(self, 
            test_size = 0.2, 
            random_state = 2020,
            verbose = True, 
            model = None,
            model_pars = {'positive': True,
                        'alpha': 0.1}):
        
        '''
        Fits the training data on the model.
        
        Args:
            test_size (float): between 0-1, fraction of the test data
            random_state (int): random state 
            verbose (bool): True to verbose, False to not verbose
            model (string): "LinearRegression", "Lasso", "Ridge" or a model function.
            model_pars (dict): dictionary of model parameters
        
        Returns:
            model object
        '''
        
        if model is None:
            model = self.model
            
        train_data_processed = self.__preprocessing(self.train_data)

        #identify colienar vairables
        if self.drop_colinears:
            self.included_variables = drop_colinearity(train_data_processed)
        else:
            self.included_variables = list(range(train_data_processed.shape[1]))
        
        features = train_data_processed.iloc[:, self.included_variables]
        
        self.preprocessed_feature = features
        
        if features is None:
            return
        
        labels = self.train_data[self.columns['label']]
        
        X_train, X_test, y_train, y_test = train_test_split(features, 
                                                            labels,
                                                            test_size = test_size, 
                                                            random_state = random_state)
        
        if model == 'LinearRegression':
            model = LinearRegression(
                normalize = True,
                #positve = model_pars['positive'] # not working in Apple M1 scikit-learn
            )
            
        elif model == 'Lasso':
            model = Lasso(
                normalize = True,
                #positive= model_pars['positive'],  # not working in Apple M1 scikit-learn
                         alpha = model_pars['alpha'])
            
        elif model == 'Ridge':
            model = Ridge(
                normalize = True,
                #positive= model_pars['positive'],   # not working in Apple M1 scikit-learn
                         alpha = model_pars['alpha'])

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        
        y_test_pred = model.predict(X_test)

        model_summary = {'train': {'R²': r2_score(y_train, y_train_pred),
                                        'MAE': mean_absolute_error(y_train, y_train_pred),
                                        'rMSE': mean_squared_error(y_train, y_train_pred)**0.5},
                              
                              'test': {'R²': r2_score(y_test, y_test_pred),
                                       'MAE': mean_absolute_error(y_test, y_test_pred),
                                      'rMSE': mean_squared_error(y_test, y_test_pred)**0.5}
                             }
        
        self.model = model
        
        self.model_summary = model_summary

        if verbose:
            print(self.model_summary)

        return {'model': model,
                'model_summary': model_summary}

    def predict(self, df):
        '''
        Predicts the labels on a training or holdout data.
        
        Args:
            df (dataframe): with specific format
        
        Returns:
            vector of predicted labels
        '''
        
        if df is None: 
            print('input data is None')
            return
        
        features = self.__preprocessing(df)
        features = features.iloc[:, self.included_variables]
        
        if features is None:
            return 

        model = self.model
            
        labels_pred = model.predict(features)
        
        return labels_pred