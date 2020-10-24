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
defaultDB = PyMongo(app, uri = "mongodb://localhost/test")
       
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

@app.route("/get_scenario2_data", methods=["GET"])
def get_scenario2_data():
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




@app.route("/predictSuspectType", methods=["POST"])
def predict_suspect_type():
    print("herer")
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





@app.route("/mongodb/collection", methods=["GET"])
def get_collection_data():
    # Please rememeber an already user db connection information must exists in our mongo db so that we can make connection and request required Database
    connect_db = request.args.get('user_db')
    connect_username = request.args.get('username')
    connect_collection = request.args.get('collection')
    try:
        myDocument = defaultDB.db.user_db_info.find({"username":connect_username})
        connect_URL = ""
        for connect_data in myDocument[0]['db_infos']:
            if(connect_db == connect_data['db_database']):
                connect_URL = "mongodb://"+ connect_data["db_username"] +":"+ connect_data["db_password"] +"@" + connect_data["db_ipaddress"] +"/"+ connect_data["db_database"] + "?authSource="+ connect_data["db_authSource"]
                break
        if connect_URL:
            mongo1 = PyMongo(app, uri = connect_URL)
            user_documents =  mongo1.db[connect_collection].find({})

            # keys = []
            # TODO: Separate All keys and sub keys dynamically in dataset or documents before sending them for preprocessing etc.
            # for doc in user_documents:
            
            #     for key, val in doc.items():
            #         print(key +" \n")
            #         print(val)
            #         if type(val) is dict:
            #             print("is dict")
            #         if type(val) is list:
            #             for x in val:
            #                 if type(x) is dict:
            #                     print (x)
            #                     print("object inside of array list")

            data_santized = json.loads(json_util.dumps(user_documents))
            return jsonify(
                data = data_santized
            ) 

        else:
            return jsonify(
                msg="Error connecting to DB"
            )

    except:
        return jsonify(msg="no collection information found",msgType="error")


    




@app.route("/file-upload", methods=["POST"])
def file_request():
    if request.method == "POST":
        myFile = request.files['input_file']
        ## defining types of files to accept
        accept_files = [
            "text/csv","application/json","text/plain",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
            ]
        if(myFile):
            if (myFile.content_type in accept_files):
                # TODO perform preprocessing on data like filtering out csv columns etc
                #upload file
                msg = upload_file(myFile)
                if msg['msgType'] == "success":
                    myFileLoc = msg['fileLocation']
                    my_dataframe = pd.read_csv(myFileLoc)
                    
                    json_data = json.loads(my_dataframe.to_json(orient='records'))

                    return jsonify({"csv_data":json_data})
                else:
                    jsonify({"msg":"unable to read"})

            else:
                return jsonify(
                    msg="Invalid File type provided! "
                )
        else:
            return jsonify(
                msg = "Error uploading"
            )


app.run()