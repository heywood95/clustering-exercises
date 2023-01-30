#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_mall_data():
    '''
    This function reads the mall data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = """
                SELECT *
                FROM customers;
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    
    return df

def acquire_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('mall_customers.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('mall_customers.csv', index_col=0)
        
    else:

        #creates new csv if one does not already exist
        df = get_mall_data()
        df.to_csv('mall_customers.csv')

    return df

def prep_mall(df):
    '''Prepares acquired mall data for exploration'''
    
    # Detect outliers using IQR
    q1, q3 = dataframe[column].quantile([0.25, 0.75])
    
    iqr = q3 - q1
    
    upper_bound = q3 + 1.5 * iqr
    
    np.where(dataframe[column] > upper_bound, 1, 0)
    
    lower_bound = q1 - 1.5 * iqr
    
    np.where(dataframe[column] < lower_bound, 1, 0)

    my_list = ['age', 'annual_income', 'spending_score']

    for col in my_list:
    
        mall_df[f'{col}_upper_outliers'] = upper_outlier_detector(mall_df, col)
        mall_df[f'{col}_lower_outliers'] = lower_outlier_detector(mall_df, col)
        
    # Get dummies for non-binary categorical variables
    dummy_df = pd.get_dummies(df[['gender']], dummy_na=False)
    
    # Concatenate dummy dataframe to original 
    df = pd.concat([df, dummy_df], axis=1)
    
    # Handel missing data
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    
    # Split the data
    train_validate, test = train_test_split(mall_df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    return train, validate, test

def scale_data(train, 
               validate, 
               test, 
               columns_to_scale=['age', 'annual_income', 'spending_score'], return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    from sklearn.preprocessing import MinMaxScaler
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    scaler = MinMaxScaler()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled

