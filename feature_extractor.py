from keras.preprocessing import image
from keras.models import Model
from numpy import asarray
from keras.applications.vgg16 import VGG16 , preprocess_input
import numpy as np
class FeatureExtractor:
    def __init__(self):
        base_model = VGG16(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self , img):
        img = img.resize((224, 224)) 
        img = img.convert('RGB')
        x = asarray(img) 
        x = np.expand_dims(x, axis=0)  
        x = preprocess_input(x)
        feature = self.model.predict(x)[0] 
        return feature / np.linalg.norm(feature) 
    

