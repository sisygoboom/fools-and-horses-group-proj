# -*- coding: utf-8 -*-
import joblib

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
        
        
    

