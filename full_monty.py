# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:21:44 2020

@author: chris
"""

from model import Model
import webbrowser, os, API

train_fresh = input('Do you want to train the model now? [y/n] ')

if train_fresh == 'y':
    print("Instantiating model...")
    modelObj = Model(stop_words=False)
    
    print("Training model...")
    modelObj.train()
    
    print("Running tests...")
    for k, v in modelObj.test().items():
        print(k + ': ' + str(v))
        
    print("Pickling the model...")
    modelObj.ml.zipIt(modelObj.get_pipe(), './Models/demonstration')
        
else:
    modelObj = Model(model_path='./Models/demonstration.gz')


body = input("Enter a body of text here for genre prediction:\n")
print(modelObj.predict_plus_accuracy(body))

print("Opening GUI page...")
webbrowser.open('file://' + os.path.realpath("index.html"))

print("Starting API...")
API.app.run(debug=False)