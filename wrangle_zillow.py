#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_connection(db, user=username, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_zillow_data():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    # Create SQL query.
    sql_query = """
            SELECT *
        FROM properties_2017 
        JOIN predictions_2017 USING(parcelid)
        JOIN propertylandusetype USING(propertylandusetypeid)
        WHERE transactiondate >= '2017-01-01' AND transactiondate <= '2017-12-31' 
        AND latitude != 'NULL' AND longitude != 'NULL' 
        AND propertylandusedesc = 'Single Family Residential';
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def acquire_zillow():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:

        #creates new csv if one does not already exist
        df = get_zillow_data()
        df.to_csv('zillow.csv')

    return df

def prep_zillow(df, prop_required_column, prop_required_row):
    '''Prepares acquired zillow data for exploration'''
    
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df

