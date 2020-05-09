# fools-and-horses-group-proj
Group project for software engineering practice - Northumbria University.

### Following the pep-8 python stlying standard
[Read more here](https://www.python.org/dev/peps/pep-0008/)

## Installation

##### Conda:
Import the environment used for running our code:

`conda env create -f environment_droplet.yml`

Run the demo script through the environment:

```
conda activate ai_gproj
python full_monty.py
```

##### Python/PIP:
Import the dependancies:

`pip install -U flask`

`pip install -U flask-cors`

`pip install -U sklearn`

`pip install -U pandas`

`pip install -U joblib`

Run the demo script:

`python full_monty.py`


## Use

---

#### Instantiation

##### Instantiate a model
```
from model import Model

mdl = Model() # uses default dataset (/.Data/balancedData.pickle)
```

##### Instantiate a model using a custom dataset
```
from model import Model

mdl = Model(dataset_path="/PATH/TO/DATA.pickle")
```
note: dataset must be a pickled dataframe containing "Plot" and "Genre" fields

##### Instantiate a model class using a pre-trained (pickled) pipeline
```
from model import Model

mdl = Model(model_path="/PATH/TO/MODEL.gz)
```
note: to `.test()` a model imported this way, you must include the data it was trained with in the `dataset_path` parameter, however predictions work fine without the original dataset

#### Other instantiation parameters
- stop_words - Use standard english stop words *(defaults to false)*
- exclude_fnames - Use first names as stop words as well *(defaults to true)*
- test_size - ratio of data to be used for testing *(defaults to 0.2 out of a range of 0-1)*

---

#### Train a pipeline (using the dataset from instantiation)
```
mdl.train()
```

---

#### Store the pipeline
```
pipe = mdl.get_pipe()

# to save the pipe for use later
mdl.ml.zipIt(pipe, "/PATH/PIPE_NAME")
```

---

#### Test model
```
td = mdl.test()
print(td)
```

---

#### Predict

##### Standard prediction from user input
```
text_to_predict = input("input: ")

prediction = mdl.predict_custom(text_to_predict")
print(prediction)
```

##### Make a prediction and return the % accuracy of that prediction
```
text_to_predict = input("input: ")

prediction, accuracy = mdl.predict_custom(text_to_predict")
print(prediction)
print(accuracy)
```

##### Make predictions from the test data
```
predictions = mdl.predict_test_data()
print(predicitions)
```
note: pipeline must be trained with the same dataset passed in instantiation