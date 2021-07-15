# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''Load data from SQL lite DB
    
    INPUT: 
    database_filepath - path to database 

    OUTPUT: 
    Target and Features for use in model.  
    '''

    # load data from database
    location = 'sqlite:///' + database_filepath
    engine = create_engine(location)
    df = pd.read_sql_table('Disastor_Recovery',engine)

    X = df['message'] #feature 
    Y = df.drop(['message','id','original','genre'], axis = 1 ) #multioutput target
    y_labels = Y.columns #column names
    return X, Y, y_labels


def tokenize(text):
    ''' Tokenize and cleanse messages column
    
    INPUT: 
    text - list of messages for input

    OUTPUT: 
    clean_tokens - clean messages list.  
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    ''' build out model using grid search and pipeline    
    INPUT: 
    [none]

    OUTPUT: 
    cv = grid searched model.   
    '''
# build out pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=1)))
                    ])
    parameters =  {'clf__estimator__n_estimators': [10,20,30]
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv 


def evaluate_model(model, X_test, Y_test, y_labels):
    ''' build out model using grid search and pipeline 
    
    INPUT: 
    

    OUTPUT: 
      
    '''
    
    # Predict results using the newly optimised parameters 
    y_pred_rev = model.predict(X_test)
    y_pred__rev_df = pd.DataFrame(y_pred_rev, columns = y_labels)

    #Look for accuracy
    for ind, columns in enumerate(Y.columns): 
        print(columns)
        print(classification_report(Y_test.iloc[:,ind], y_pred_df.iloc[:,ind]))
    
    return 


def save_model(model, model_filepath):
    # Save the Model to file
    Model_Filename = 'MCC_Model.pkl'  

    with open(Model_Filename, 'wb') as file:  
        pickle.dump(model, file)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()