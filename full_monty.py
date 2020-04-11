# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 23:21:44 2020

@author: chris
"""

from model import Model
import webbrowser, os, API

print("Instantiating model...")
mdl = Model()
print("Training model...")
mdl.train()
print("Running tests...")
mdl.test()
print("Pickling the model...")
mdl.ml.zipIt(mdl.get_pipe(), './Models/demonstration')

print("Opening GUI page...")
webbrowser.open('file://' + os.path.realpath("index.html"))

print("Starting API...")
API.app.run(debug=True)