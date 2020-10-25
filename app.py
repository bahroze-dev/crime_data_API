from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_pymongo import PyMongo
from bson import ObjectId, json_util
import json
import os



app=Flask(__name__)
app.config["DEBUG"]=True
CORS(app)

## DB
defaultDB = PyMongo(app, uri = "mongodb+srv://client:hacker101@cluster0.bqwku.mongodb.net/test?retryWrites=true&w=majority")
       
## Loading up ML models and imports
import pickle
from datetime import datetime
from faker import Faker
fake = Faker()

# Load class Model back from file
Pkl_Filename = "./models/Pickle_KNN_CRIME_CLASS_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    Pickle_KNN_CRIME_CLASS_Model = pickle.load(file)

# Load regress Model back from file
Pkl_Filename = "./models/Pickle_SVR_CRIME_REGRESS_Model.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    Pickle_SVR_CRIME_REGRESS_Model = pickle.load(file)

## Scaler for scaling our inputs 

Pkl_Filename = "./models/Pickle_SVR_CRIME_REGRESS_Scaler.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    Pickle_SVR_CRIME_REGRESS_Scaler = pickle.load(file)

Pkl_Filename = "./models/Pickle_KNN_CRIME_CLASS_Scaler.pkl"  

with open(Pkl_Filename, 'rb') as file:  
    Pickle_KNN_CRIME_CLASS_Scaler = pickle.load(file)

##
## importing recomendation matrix of data
import numpy
dict_data = numpy.load('./models/cosine_matrix.npz')
# extract the first array
cosine_sim = dict_data['arr_0']

##


@app.route("/", methods=["GET"])
def home_page():
    return render_template('sample_front.html')


@app.route("/getData", methods=["GET"])
def get_crime_data():
    myDocument = defaultDB.db.crime.find({},{"_id":0,"Local_Area_Cat":0,"OFNS_DESC_Cat":0,"PERP_RACE_Cat":0,"PERP_SEX_Cat":0})
    data_santized = json.loads(json_util.dumps(myDocument))
    return jsonify(
        data = data_santized
    ) 

@app.route("/get_scenarios_data", methods=["GET"])
def get_scenarios_data():
    all_offenses = defaultDB.db.crime.distinct("OFNS_DESC")
    all_genders = defaultDB.db.crime.distinct("PERP_SEX")
    all_Local_Area = defaultDB.db.crime.distinct("Local Area")
    
    data_santized_offenses = json.loads(json_util.dumps(all_offenses))
    data_santized_genders = json.loads(json_util.dumps(all_genders))
    data_santized_area = json.loads(json_util.dumps(all_Local_Area))

   
    return jsonify(
        all_offenses = data_santized_offenses,
        all_genders = data_santized_genders,
        all_Local_Area = data_santized_area,  
    ) 

@app.route("/get_scenario2_predicted", methods=["GET"])
def get_scenario2_predicted():
    all_predicted = defaultDB.db.predicted_data.find({}).sort('Time',-1)
    
    data_santized_predicted = json.loads(json_util.dumps(all_predicted))
    
   
    return jsonify(
        all_predicted = data_santized_predicted, 
    ) 


## recommendation of suspects

@app.route("/get_recommended_suspects", methods=["GET"])
def get_recommended_suspects():

    get_suspects_recommended = defaultDB.db.predicted_suspects.find({}).sort('Time',-1)
    
    data_santized_recommended = json.loads(json_util.dumps(get_suspects_recommended))
    
    return jsonify(
        recommended_suspects = data_santized_recommended[0], 
    ) 






#scenario one predicted
@app.route("/recommendsSuspectList", methods=["POST"])
def recommendsSuspectList():
    req_data = request.get_json()
    
    crime_type = req_data['crimeType']
    location = req_data['location']
    
    
    location_cat = defaultDB.db.crime.find_one({'Local Area':location},{'Local_Area_Cat':1})
    location_cat = location_cat['Local_Area_Cat']
    

    suspects_list = defaultDB.db.crime.find(
        {'OFNS_DESC':crime_type,"Local_Area_Cat":location_cat},
        {"index":1,"Name":1,"Age":1,"PERP_RACE":1,"PERP_SEX":1,"_id":0}
    )

    data_santized_suspects_list = json.loads(json_util.dumps(suspects_list))


    recommendation_list = []
    for docs in data_santized_suspects_list:
        name = docs["Name"]
        index = docs["index"]
        recommendation_list.append(docs)
        recommendation_suspects = get_recommendations(name,cosine_sim,index)
        for i in recommendation_suspects:
            document_suspect = defaultDB.db.crime.find_one({'index':i},{"Name":1,"Age":1,"PERP_RACE":1,"PERP_SEX":1,"_id":0})
            recommendation_list.append(document_suspect)
    
    ## This scenario is considered if the crime has never been commited in this region so we take crime only as input
    ## and check features of similiar suspects with everyone else and top are listed
    if not recommendation_list:
        suspects_list = defaultDB.db.crime.find(
            {'OFNS_DESC':crime_type},
            {"index":1,"Name":1,"Age":1,"PERP_RACE":1,"PERP_SEX":1,"_id":0}
        )
        data_santized_suspects_list = json.loads(json_util.dumps(suspects_list))
        for docs in data_santized_suspects_list:
            name = docs["Name"]
            index = docs["index"]
            recommendation_list.append(docs)
            recommendation_suspects = get_recommendations(name,cosine_sim,index)
            for i in recommendation_suspects:
                document_suspect = defaultDB.db.crime.find_one({'index':i},{"Name":1,"Age":1,"PERP_RACE":1,"PERP_SEX":1,"_id":0})
                recommendation_list.append(document_suspect)
        


    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    mydict = { "recommended_suspects":recommendation_list ,"Time": dt_string }
    x = defaultDB.db.predicted_suspects.insert_one(mydict)


   
    return jsonify(
        msg = "success", 
    ) 

def get_recommendations(name, cosine_sim=cosine_sim, index=0):
    # Get the index of the suspects that matches the name
    idx = int(index)

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:3]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movie_indices



@app.route("/predictSuspectType", methods=["POST"])
def predict_suspect_type():
    req_data = request.get_json()
    
    crime_type = req_data['crimeType']
    location = req_data['location']
    gender = req_data['gender']
    
    
    crime_type_cat = defaultDB.db.crime.find_one({'OFNS_DESC':crime_type},{'OFNS_DESC_Cat':1})
    location_cat = defaultDB.db.crime.find_one({'Local Area':location},{'Local_Area_Cat':1})
    gender_cat = defaultDB.db.crime.find_one({'PERP_SEX':gender},{'PERP_SEX_Cat':1})


    crime_type_cat = crime_type_cat['OFNS_DESC_Cat']
    location_cat = location_cat['Local_Area_Cat']
    gender_cat = gender_cat['PERP_SEX_Cat']
    

    ## Randomly generating date and time since for current situation user isn't specifying
    
    date_time = str(fake.date_time_between_dates(datetime_start=datetime(2020,1,21,11,00,00),datetime_end=datetime(2020,10,22,11,00,00)))
    date =  date_time.split(" ")[0]
    time =  date_time.split(" ")[1]
    
    year =  int(date.split("-")[0])
    month =  int(date.split("-")[1])
    day =  int(date.split("-")[2])
    
    
    hours_24 =  int(time.split(":")[0])
    minutes =  int(time.split(":")[1])
    seconds =  int(time.split(":")[2])
    

    input_data_class = [location_cat,crime_type_cat,gender_cat,year,month,day,hours_24,minutes,seconds]
    


    ##
    
    ## Pattern of features for classification of race.

    # 	Local_Area_Cat	OFNS_DESC_Cat	PERP_SEX_Cat	Year	Month	Day	Hours(24)	Minutes  Seconds

    scaled_input_data_class = Pickle_KNN_CRIME_CLASS_Scaler.transform([input_data_class])
    
    predicted_race = Pickle_KNN_CRIME_CLASS_Model.predict(scaled_input_data_class)
    print(predicted_race[0])

    ##
    ## Pattern of features for regression of age prediction
    # Local_Area_Cat	OFNS_DESC_Cat	PERP_RACE_Cat	PERP_SEX_Cat	Year	Month	Day	Hours(24)	Minutes Seconds

    input_data_reg = [location_cat,crime_type_cat,predicted_race[0],gender_cat,year,month,day,hours_24,minutes,seconds]

    scaled_input_data_reg = Pickle_SVR_CRIME_REGRESS_Scaler.transform([input_data_reg])

    predicted_age = Pickle_SVR_CRIME_REGRESS_Model.predict(scaled_input_data_reg)

    predicted_race = predicted_race.tolist()
    predicted_age = predicted_age.tolist()


    predicted_race = defaultDB.db.crime.find_one({'PERP_RACE_Cat':predicted_race[0]},{'PERP_RACE':1})
    predicted_race = predicted_race['PERP_RACE']

    ## Saving the data to DB as well
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    mydict = { "Age": round(predicted_age[0],0), "Race": predicted_race, "Time": dt_string }

    x = defaultDB.db.predicted_data.insert_one(mydict)

    return jsonify(
        Age = round(predicted_age[0],0),
        Race =  predicted_race
    ) 

app.run(host='0.0.0.0', port=4000)