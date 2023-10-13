import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class LocalFeature:
    def __init__(self, feature_type, cluster_size, test_size = 0.25, image_type = 'original'):
        self.feature_type = feature_type
        self.cluster_size = cluster_size
        self.test_size = test_size
        self.image_type = image_type
    
    def sift_features(self, image_array):
        features_list = []
        labels = []
        sift = cv2.SIFT_create()
        for idx, itr in enumerate(image_array):
            if self.image_type == 'original':
                temp = cv2.cvtColor(itr, cv2.COLOR_BGR2GRAY)
            else:
                temp = itr
            eight_bit_image = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            keypoints, descriptors = sift.detectAndCompute(eight_bit_image, None)
            if(descriptors is not None):
                features_list.append(descriptors)
                if idx < 13779 :
                    labels.append(1)
                else:
                    labels.append(0)
        return features_list, labels
    
    def kaze_features(self, image_array):
        features_list = []
        labels = []
        kaze = cv2.KAZE_create()
        for idx, itr in enumerate(image_array):
            if self.image_type == 'original':
                temp = cv2.cvtColor(itr, cv2.COLOR_BGR2GRAY)
            else:
                temp = itr
            eight_bit_image = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            keypoints, descriptors = kaze.detectAndCompute(eight_bit_image,None)
            if(descriptors is not None):
                features_list.append(descriptors)
                if idx < 13779 :
                    labels.append(1)
                else:
                    labels.append(0)
        return features_list,labels
    
    def extract_features(self, image_array):
        if self.feature_type == 'SIFT':
            features_list, labels = self.sift_features(image_array)
        elif self.feature_type == 'KAZE':
            features_list, labels = self.kaze_features(image_array)
        return features_list, labels
    
    def creating_cluster(self, feature_list):
        lis = []
        for i in range(len(feature_list)):
            for f in range(feature_list[i].shape[0]):
                lis.append(feature_list[i][f])
        x_image_features = np.array(lis)
        x_image_scaled = StandardScaler().fit_transform(x_image_features)
        kmeans = KMeans(n_clusters = self.cluster_size, random_state = 0).fit(x_image_scaled)
        return kmeans
    
    def create_bag_of_words(self, feature_list, kmeans):
        bovw_v = np.zeros([len(feature_list), self.cluster_size])
        for index, features in enumerate(feature_list):
            for i in kmeans.predict(features):
                bovw_v[index, i] +=1
        return bovw_v
    
    def fit(self, image_array):
        features_list, labels = self.extract_features(image_array)
        print(len(features_list), len(labels))
        x_train, x_test, y_train, y_test = train_test_split(features_list, 
                                                                labels, test_size = self.test_size, random_state = 0)
        
        kmeans = self.creating_cluster(x_train)
        bow_v = self.create_bag_of_words(x_train, kmeans)
        bow_v_test = self.create_bag_of_words(x_test, kmeans)
        return bow_v, bow_v_test, np.array(y_train), np.array(y_test)
        