# import libraries
import pandas as pd 
from sqlalchemy import create_engine
import sys

# function to load datafiles 
def load_data(messages_filepath, categories_filepath):
    '''Load data from specified filepaths for processing
    
    INPUT: 
    messages_filepath - messages file
    categories_filepath - categories file

    OUTPUT: 
    df - merged dataframe of messages & categories 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id')
    return df 
   
 #function to conduct data cleansing  
def clean_data(df):
    '''Process data     
    INPUT: 
    df - merged dataframe from load_data()

    OUTPUT: 
    df2 - cleansed dataframe  
    '''
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[1]
    category_colnames = list(map(lambda x:x.split('-')[0] , row))
    categories.columns = category_colnames

    #isolate value only
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    #drop original categories column 
    df.drop('categories',axis=1,inplace=True)

    # concatonate revised categories back to orginal df
    df2 =  pd.concat([df,categories],axis=1)
    df2.drop_duplicates(inplace=True)
    print(df2['related'].unique())
    # Remove 2 values from the 'Related' column
    df2['related'] = df2['related'].replace(2,1)
    print(df2['related'].unique())
    return df2

#function to save data to new SQL lite DB
def save_data(df2, database_filename):
    '''Save data to SQL lite DB
    
    INPUT: 
    df2 - cleansed dataset

    OUTPUT: 
    SQl database containing cleansed data.  
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df2.to_sql('Disastor_Recovery', engine, index=False, if_exists = 'replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample:  '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()