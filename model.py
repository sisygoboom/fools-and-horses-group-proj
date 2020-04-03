import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings
import dataLoader
warnings.filterwarnings('ignore')


# load data
def load_data():
    l = dataLoader.Dataset()
    data = l.load_stored()
    return data


def load_dataset():
    with open('Data/firstnames.txt') as f:
        firstnames = f.read().splitlines()
    return firstnames


def get_data_head(data, lines):
    data.head(lines)
    print(data.Genre)


def get_data_shape(data):
    print(data.shape)


def get_data_dtypes(data):
    print(data.dtypes)


# split data into training and test sets
def split_test_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data.Plot, data.Genre, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# create list of stop words
def create_stop_words(firstnames):
    stop_words = ENGLISH_STOP_WORDS.union(firstnames)
    return stop_words


# make the pipeline
def make_pipeline(stop_words):
    text_pipe = Pipeline([
        ('vect', TfidfVectorizer(stop_words=stop_words, lowercase=True)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    text_pipe.fit(data.Plot, data.Genre)
    return text_pipe


# print types of genre avaialable
def print_genre_types(text_pipe):
    print(text_pipe.named_steps['clf'].classes_)


# for each genre type display 10 features with the biggest classification weights
def display_top_features(text_pipe):
    feature_names = text_pipe.named_steps['vect'].get_feature_names()
    classifier_weights = text_pipe.named_steps['clf'].coef_
    classifier_classes = text_pipe.named_steps['clf'].classes_

    for i, genre in enumerate(classifier_classes):
        top10 = np.argsort(-classifier_weights[i])[:10]
        print(genre)
        print([feature_names[j] for j in top10])


# Prediction test
def prediction_test(X_test, text_pipe):
    print(text_pipe.predict(X_test))


# Evaluate the performance on test set
def evaluate_performance_test_set(X_train, X_test, y_train, y_test, text_pipe):
    grid_params = {
        'vect__stop_words': [None, 'english'],
        'tfidf__use_idf': (True, False),
    }
    search = GridSearchCV(text_pipe, grid_params)
    search.fit(X_train, y_train)
    print(search.best_params_)
    print(search.score(X_test, y_test))
    print(search.error_score, search.best_estimator_, search.best_score_)


# false prediction
# text = [
#     "Gang kill police killed father man men brother family kills",
# ]


def predict_text(text):
    print(text_pipe.predict(text))


# most frequent genres different than unknown
def plot_frequent_genres(data):
    freq_gen = data[data.Genre != "unknown"]

    plt.figure(figsize=(12,6))
    plt.title('15 most frequent genre types', fontsize = 20)
    plt.xlabel('Genre', fontsize = 20)
    plt.ylabel('Count', fontsize = 20)
    sns.countplot(freq_gen.Genre, order = pd.value_counts(freq_gen.Genre).iloc[:15].index, palette = sns.color_palette("Blues_d", 15))
    plt.xticks(size=13, rotation=90)
    plt.yticks(size=13)
    sns.despine()
    plt.show()


# evaluate the performance on training set
# calculate the number of correctly and incorrectly classified examples
def evaluate_performance_training_set(text_pipe, X_train, y_train):
    print(metrics.confusion_matrix(y_train, text_pipe.predict(X_train)))


# calculate the precision, recall and f1-score
def get_f1_score(text_pipe, X_train, y_train):
    print(metrics.classification_report(y_train, text_pipe.predict(X_train)))

