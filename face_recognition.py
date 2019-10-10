# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 01:59:56 2019

@author: Adil
"""

import dlib
import numpy as np
import time
import face_recognition
import os
from PIL import Image
from matplotlib import pyplot as plt

def get_face_embeddings_from_image(image, convert_to_rgb=False):
   
    if convert_to_rgb:
        image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_locations, face_encodings

class Pic:
    
    def __init__(self,path):
        self.path = path
        self.comparisons={}
        
    def load(self):
        self.img = face_recognition.load_image_file(path)
        
    def embed(self):
        self.locations,self.embedding=get_face_embeddings_from_image(self.img)
    
    def compare(self,other):
        comparison=face_recognition.face_distance(other.embedding,self.embedding[0])
        print('{0:<20}{1:<20}{2:>10}'.format(self.path,other.path,str(comparison[0])))
        self.comparisons[other.path]=comparison[0]
    
    def anchor(self):
        self.anchors = face_recognition.face_landmarks(self.img,self.locations)  
    
    def plot(self,ax):
        anchor = self.anchors[0]      
        for key in anchor.keys():
            for pair in anchor[key]:
                ax.plot(pair[0],-pair[1],'o', color='black')
                ax.set_title(self.path)
    
    def display(self):
        image=face_recognition.load_image_file(self.path)
        top, right, bottom, left = self.locations[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        print(top, right, bottom, left)
        pil_image.show()
        