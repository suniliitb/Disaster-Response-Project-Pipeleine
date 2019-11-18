import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pandas as pd
import numpy as np 
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def load_data(database_filepath):
    '''
    Function to load the sql database using the database filepath and split the dataset features X dataset and target Y dataset.
    Returns X and Y datasets alongwith the category names
    '''
    
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("messages_disaster", con = engine)
    X = df["message"]
    Y = df.drop(["message", "genre", "id", "original"], axis = 1)
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
   '''
   Function to tokenize the text messages.
   Inputs the complete text messages and returns clean tokenized text as a list
   '''

   tokens = word_tokenize(text)
   lemmatizer = WordNetLemmatizer()

   clean_tokens = []
   for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
   return clean_tokens


def build_model():
    '''
    Function to build the model by creating an ML pipeline and use a set of parameters to further tune the model by applying Grid Search
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
   
    parameters = {'clf__estimator__max_depth': [5, 10, None]} 

    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function to test the model for each category using accuracy scores
    '''
    y_pred = model.predict(X_test)
    # print classification report
    for i, col in enumerate(category_names):
        print (col)
        print(classification_report(Y_test[col], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    Function to save the model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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