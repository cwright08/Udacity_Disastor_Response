# Disaster Response Pipeline Project
Respository for Disastor Response machine learning pipleine and interactive dashboard site, compelted as part of Udacity Nanodegree in Data Science using real world data provided by Figure8. 

### Project Description & Motivation 


### User instructions 
1. In the project's root directory run the following commands.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ on your machine the view results 

### Sample Image ### 
![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)


### Respository Guide
-app
    --run.py
    --templates
        --go.html
        --master.html
-data
    --disaster_categories.csv
    --disaster_messages.csv
    --DisasterResponse.db
    --process_data.py
-models
    --train_classifier.py

### Acknowledgements & Citations: 
This project is built using templates provided by Udacity for their Data Science Nanodegree and code from the lesson 'Data Engineering' is used extensively throughout. 

All data courtesy of Figure Eight (https://www.linkedin.com/company/figureeight/) via Udacity. 

### Libraries & Technology
- Python 3.8 on Anaconda
- Pandas
- Matplot lib
- SckitLearn
- NLTK
- Flask
- Plotly

### Website Screenshot
