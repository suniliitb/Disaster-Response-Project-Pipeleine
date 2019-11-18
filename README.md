# Disaster-Response-Project-Pipeline

### Project Overview
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight.
The initial dataset contains pre-labelled tweet and messages from real-life disaster. 
The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided in the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper databse structure
2. Machine Learning Pipeline to train a model able to classify text message in categories
3. Web App to show model results in real time. 

### Requirements
* Python 3.7
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Instructions to run the app:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to create your web app.
    `python run.py`

3. Go to http://0.0.0.0:3002/

### Files of importance:

   * `app/templates/*` templates/html files for web app
   * `data/process_data.py`Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing
   data in a SQLite database
   * `models/train_classifier.py` A machine learning pipeline that loads data, trains a model, and saves the trained
   model as a .pkl file for later use
   * `run.py` The file used to launch the Flask web app used to classify disaster messages
