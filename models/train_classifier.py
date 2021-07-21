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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle


def load_data(database_filepath):
    '''Load data from SQL lite DB and create features and target dataframes
    
    INPUT: 
    database_filepath - path to database 

    OUTPUT: 
    Target and Features for use in model.  
    '''

    # load data from database
    file_location = 'sqlite:///' + database_filepath
    engine = create_engine(file_location)
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
    [NONE]

    OUTPUT: 
    cv = grid searcch model.   
    '''
# build out pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf',MultiOutputClassifier(RandomForestClassifier(random_state=1)))
                    ])
    # parameters =  {'clf__estimator__n_estimators': [1,2]}                
    parameters =  {'clf__estimator__n_estimators': [10,25,50], 
    'clf__estimator__criterion': ['gini','entropy'] 
    } #tune on number of trees and split criteria. 

# Best Parameters Are{'clf__estimator__criterion': 'gini', 'clf__estimator__n_estimators': 50}

    cv = GridSearchCV(pipeline, parameters, verbose=3)
    return cv 


def evaluate_model(model, X_test, Y_test, y_labels):
    ''' Build out model using grid search and pipeline, evaluate performance of the model. 
    
    INPUT: 
    model = trainied model
    X_test = testing set features
    Y_test = testing set targets
    y_labels = target column labels

    OUTPUT: 
    Tuned model report. 
      
    '''
    # Predict results using the newly optimised parameters 
    y_pred_rev = model.predict(X_test)
    y_pred_rev_df = pd.DataFrame(y_pred_rev, columns = y_labels)

    #Look for accuracy
    for ind, columns in enumerate(y_pred_rev_df.columns): 
        print(columns)
        print(classification_report(Y_test.iloc[:,ind], y_pred_rev_df.iloc[:,ind]))
    pass 


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
        para_selected = model.best_params_
        print("Best Parameters Are" + str(para_selected))

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