from flask import Flask, jsonify, request
from flask_cors import CORS
import modelLoader as fl
from model import Model

san = fl.inputSan()
load = fl.fileManager()

#create the flask app
app = Flask(__name__)

cors = CORS(app)

#load the trained model from file 
model = Model(model_path="./Models/demonstration.gz")

#predict is the end point, decorator
@app.route('/prediction', methods=["POST"])
def predict_genre():
    if request.method == 'POST':
      #get the movie plot from the post request  
        movie_plot = request.form['plot']
        
        #check if movie plot is empty: if not sanitise data
        if san.length_check(movie_plot):

            movie_plot = san.sanitise_input(movie_plot)
            #check length again, if length after sanitisation is not empty then make a prediction
            if san.length_check(movie_plot):
                #make prediction on sanitised movie plot
                pred, accuracy = model.predict_plus_accuracy(movie_plot)
                
                #convert from numpy array to list 
                pred = pred.tolist()
                
            else:
                #if the santised plot is empty set prediction equal to error to send to GUI
                pred = ["<h1 style='color: Red;' >ERROR: Invalid movie plot entered</h1>\n<p>Please enter a movie plot for prediction</p>"]
                accuracy = ""
        else: 
            #if empty movie plot is entered set plot equal to an error message to send to the GUI
            pred = ["<h1 style='color: Red;' >ERROR: Empty move plot entered </h1>\n<p>Please enter a movie plot for prediction</p>"]
            accuracy = ""
          
        #create a json object named response containing the prediction and accuracy    
        response = jsonify({'genre_prediction': pred, 'accuracy': accuracy})
        
        response.headers.add('Access-Control-Allow-Origin', '*')
        #return json object response
        return response



if __name__ == "__main__":
    app.run(debug=True) 
    

