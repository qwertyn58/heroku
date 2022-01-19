import pickle
from flask import Flask,jsonify,request
import numpy as np
import pandas as pd

app = Flask(__name__)
#getting our trained model from a file we created earlier
model = pickle.load(open("modele.pkl","rb")) 
seuil = pickle.load(open("seuil.pkl","rb"))
df_clients = pd.read_csv('df_clients.csv')

#defining a route for only post requests
@app.route('/predict', methods=['GET'])
def predict():

    #getting an array of features from the post request's body
    sk_id = request.args
    if int(request.args['sk_id']) in df_clients['SK_ID_CURR'].values:
        feature_array = df_clients[df_clients['SK_ID_CURR']==int(request.args['sk_id'])].iloc[:,1:].values
        
        #creating a response object
        #storing the model's prediction in the object
        response = {}
        response['seuil']=seuil
        if model.predict_proba(feature_array)[0,1]>seuil:
            response['predictions'] = 1
        else:
            response['predictions'] = 0   
        #returning the response object as json
        return jsonify(response,model.predict_proba(feature_array)[0,0])
    else :
        return jsonify('identifiant incorrect')
    

if __name__ == "__main__":
    app.run(debug=True)
