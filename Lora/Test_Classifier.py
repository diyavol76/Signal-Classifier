import numpy as np
#import keras
from keras.models import load_model
from keras import backend as K
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from Lora.dataset_preparation import ChannelIndSpectrogram
import torch
from torch.utils.data import DataLoader

class Classifier():

    def __init__(self,):

        self.label='label'


    def Lora_Classification(self,data_register,data_inference,extractor_name):

        feature_extractor = load_model(extractor_name,custom_objects={"K": K}, compile=False)

        ChannelIndSpectrogramObj = ChannelIndSpectrogram()
        #for data, target in data_register:
         #   print("data",target)
        #print("type of data ",type(data_register))
        #data_enrol,label_enrol = data_register
        data_enrol=[]
        label_enrol=[]
        #enroll_dataloader = DataLoader(data_register, shuffle=True)

        #iter_data = iter(enroll_dataloader)
        # iter_data = torch.from_numpy(iter_data)
#        while True:
#            train_features, train_labels = next(iter_data)
            
#        print("ttt",train_features.shape)
        for t_images, t_label in data_register:

            data_enrol.append(t_images)

            label_enrol.append( t_label.numpy())
        print("shape of", np.shape(data_enrol), np.shape(label_enrol))


        #Channel independent spectogram
        data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
        print("enroll shape after spectogram",np.shape(data_enrol))


        feature_enrol = feature_extractor.predict(data_enrol)

        knnclf = KNeighborsClassifier(n_neighbors=15, metric='euclidean')
        knnclf.fit(feature_enrol, np.ravel(label_enrol))

        #data_clf,label_clf = data_inference


        data_clf=[]
        label_clf=[]
        for t_images, t_label in data_inference:
            data_clf.append(t_images)
            label_clf.append(t_label.numpy())
        print("shape of test", np.shape(data_clf), np.shape(label_clf))

        data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf)
        feature_clf = feature_extractor.predict(data_clf)

        print("enroll shape after spectogram", np.shape(data_clf))
        del data_clf

        label_predict = knnclf.predict(feature_clf)

        print(np.shape(label_enrol),np.shape(label_clf),np.shape(label_predict))
        print(np.squeeze(label_enrol))
        print(label_predict)
        print(np.squeeze(label_clf))
        label_clf=np.squeeze(label_clf)
        label_predict = np.squeeze(label_predict)
        print(np.shape(label_enrol), np.shape(label_clf), np.shape(label_predict))
        acc = accuracy_score(np.squeeze(label_clf),np.squeeze(label_predict))

        return label_predict,label_clf,acc




