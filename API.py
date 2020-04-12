from flask import Flask, jsonify, request
from flask_cors import CORS
import modelLoader as fl


load = fl.fileManager()
#create the flask app
app= Flask(__name__)

cors = CORS(app)

#load th previously trained model from file 
model = load.loadIt("./Models/version2.gz")

#predict is the end point, decorator
@app.route('/prediction', methods=["POST"])
def predict_genre():
    if request.method == 'POST':
        movie_plot = request.form['plot']
        #make the raw document data a liist 
        movie_plot = [movie_plot]
        
        #make prediction with model
        pred = model.predict(movie_plot)
        #convert from numpy array to list 
        pred = pred.tolist()
        response = jsonify('genre_prediction', pred)
        response.headers.add('Access-Control-Allow-Origin', '*')
        #send json result
        return response

#this is only used while in development, remove later
if __name__ == "__main__":
    app.run(debug=True) 
    

