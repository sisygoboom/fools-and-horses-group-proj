# -*- coding: utf-8 -*-
import joblib
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class fileManager:
   
    def pickleIt(self,data, path):
        """
        creates a pickle file of specified data
        
        param data: the file to be serialized 
        param path: the path and name for the file without pickle file extension
        
        """
        joblib.dump(data,path+ '.pickle')
        
    
    def loadIt(self,fileName):
        """
        unpickles or unzips specified data
        
        param fileName: the path and name of the file to be loaded
        return: the unpickled/unzipped file
        
        """
        return joblib.load(fileName)
    
    def zipIt(self, data, path):
        """
        compresses a file using gzip 
        
        param data: the data to be compressedd
        param path: the path and name for the file
        """
        joblib.dump(data, path + '.gz', compress = 'gzip')
        
        

#this class contains the utility functions used to sanitise input to the api
class inputSan:
    
    def sanitise_input(self, movie_plot):
        #function to sanitise the input from the GUI

        #remove stop words from the movie plot
        movie_plot = self.remove_stopwords(movie_plot)
        
        #remove contractions between words
        movie_plot = self.remove_contractions(movie_plot)
        
        #how did i miss out the return value
        return movie_plot
           
    
    
    def remove_contractions(self,movie_plot):
        #function to remove contractions between words 
        
        #remove won't and replace with will not
        movie_plot = re.sub(r"won't", "will not",str(movie_plot))
        #remove can't and replace with can not
        movie_plot = re.sub(r"can't", "can not",movie_plot)
        
        #remove 's from words and replace with is as in it's = it is
        movie_plot = re.sub(r"\'s", " is",movie_plot)
        #remove 'd from words and replace with would as in I'd = I would
        movie_plot = re.sub(r"\'d", " would",movie_plot)
        
        #remove n't from words and replace with not as in wouldn't = would not
        movie_plot = re.sub(r"n\'t", " not",movie_plot)
        #remove 're from words and replace with are as in you're = you are
        movie_plot = re.sub(r"\'re", " are",movie_plot)
        
        #remove 'll from words and replace with will as in I'll = I will
        movie_plot = re.sub(r"\'ll", " will",movie_plot)
        #remove 't from words and replace with not
        movie_plot = re.sub(r"\'t", " not",movie_plot)
        
        #remove 've from words and replace with have as in could've = could have
        movie_plot = re.sub(r"\'ve", " have",movie_plot)
        #remove 'm from words and replace with am as in I'm = I am
        movie_plot = re.sub(r"\'m", " am",movie_plot)
        
        return movie_plot
    
	
    def length_check(self, movie_plot):
        #checks if the plot is an empty string or not
        if movie_plot =="" or movie_plot == " ":
            #if empty return false
            return False
        else:
            #if not empty return true
            return True
        
    def remove_stopwords(self, movie_plot):
        #removes stop words from the movie plot 
        
        #create list of stop words 
        stop = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}
        #combine with set of already existing stop words 
        allStop = ENGLISH_STOP_WORDS.union(stop)
        #remove special characters
        movie_plot = re.sub("\S*\d\S*", "", movie_plot).strip()
        #remove all characters except A-Z/a-z
        movie_plot = re.sub('[^A-Za-z]+', ' ', movie_plot)
        #make all words lowercase
        movie_plot = ' '.join(e.lower() for e in movie_plot.split() if e.lower() not in allStop)
        
        return movie_plot

