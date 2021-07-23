# Disaster Response Pipeline Project
Respository for Disastor Response machine learning pipleine and interactive dashboard site, completed as part of Udacity Nanodegree in Data Science using real world data provided by Figure8. 

### Project Description & Motivation 
This project aimed to demonstrate how natural language processing and machine learning can be used to classify messages which are recieved in a disastor event using Python. Using real world, pre-classified data, a working web app was successfully created. The same approach could be used to classify messages in other scenarios, such as customer service. 

### User instructions 
1. In the project's root directory run the following commands.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ on your machine the view results 

### Observations and Conclusions
The project succesfully demonstrates the principles of how a natural language and machine learning model can be used to classify text. 

The project uses a fairly simple Multiclassifier model with Random Forests. This was selected after a period of research which included implementing a K-Nearest Neighbor classifier and showed fairly good performance when evaluating metrics. One problem which was clear after initial data exploration is the lack of repsonses for certain features, making it hard or impossible for models to predict the outcome for these when deploy.Further, given additional time it would be very useful to test various other models such as neural networks and more advanced NLTK methods to seek to improve performance. 


### Respository Guide
-app = files to be used to run the web app.
    --run.py
    --templates
        --go.html
        --master.html
-data = raw data, and processing files.
    --disaster_categories.csv
    --disaster_messages.csv
    --DisasterResponse.db
    --process_data.py
-models - machine learning model classifier script. 
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
![webscreenshot](Webscreenshot.jpg?raw=true)
