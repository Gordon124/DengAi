import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
class ProcessData:
    def __init__(self):
        self.data_matrix = pd.read_csv('dengue_features_train.csv')
        self.label = pd.read_csv('dengue_labels_train.csv')


    def cov_matrix(self):
        # data = np.loadtxt("dengue_features_train.csv", delimiter=',', skiprows=1, usecols=range(4, 23))
        feature_train = pd.read_csv('dengue_features_train.csv')
        rmv_nan_ft = feature_train.dropna()

        #cov_matrix = np.cov(rmv_nan_ft.transpose(),bias=True)
        keep_col = ["ndvi_ne","ndvi_nw","ndvi_se","ndvi_sw","precipitation_amt_mm","reanalysis_air_temp_k",
                    "reanalysis_avg_temp_k","reanalysis_dew_point_temp_k","reanalysis_max_air_temp_k",
                    "reanalysis_min_air_temp_k","reanalysis_precip_amt_kg_per_m2",
                    "reanalysis_relative_humidity_percent","reanalysis_sat_precip_amt_mm",
                    "reanalysis_specific_humidity_g_per_kg","reanalysis_tdtr_k","station_avg_temp_c",
                    "station_diur_temp_rng_c","station_max_temp_c","station_min_temp_c","station_precip_mm"]
        rmved_dateCol = rmv_nan_ft[keep_col]
        #why if its tranpose its 20x20
        cov_matrix = np.cov(rmved_dateCol.transpose())
        X = np.array(cov_matrix)

        # the scaler object (model)
        scaler = StandardScaler()
        scaler.fit(rmved_dateCol)
        # fit and transform the data
        scaled_data = scaler.transform(rmved_dateCol)
        #print(scaled_data)


        pca = PCA(n_components=3)
        principleComp = pca.fit_transform(scaled_data)
        #principleDf = pd.DataFrame(data=principleComp,columns=['principle component 1','principle component 2'])


        return pca.transform(scaled_data)


    def processTest(self,model):
        feature_test = pd.read_csv('dengue_features_test.csv')

        copy_test = pd.read_csv('dengue_features_test.csv')
        keep_col = ["ndvi_ne", "ndvi_nw", "ndvi_se", "ndvi_sw", "precipitation_amt_mm", "reanalysis_air_temp_k",
                    "reanalysis_avg_temp_k", "reanalysis_dew_point_temp_k", "reanalysis_max_air_temp_k",
                    "reanalysis_min_air_temp_k", "reanalysis_precip_amt_kg_per_m2",
                    "reanalysis_relative_humidity_percent", "reanalysis_sat_precip_amt_mm",
                    "reanalysis_specific_humidity_g_per_kg", "reanalysis_tdtr_k", "station_avg_temp_c",
                    "station_diur_temp_rng_c", "station_max_temp_c", "station_min_temp_c", "station_precip_mm"]

        copy_test = copy_test[keep_col]
        copy_test.to_csv('change-dengue_features_test.csv', index=False)
        copied_mean = pd.read_csv('change-dengue_features_test.csv').dropna()

        copied_test = csv.reader(open('change-dengue_features_test.csv'))

        lines = list(copied_test)

        for i in range(len(lines)):
            if i == 0:
                continue
            for j in range(len(lines[i])):
                if not lines[i][j]:
                    lines[i][j] = copied_mean[keep_col[j]].mean()


        pd.DataFrame(lines).to_csv('change-dengue_features_test.csv', header=None, index=None)
        featrue_test_filled = pd.read_csv('change-dengue_features_test.csv')

        scaler = StandardScaler()
        scaler.fit(featrue_test_filled)
        # fit and transform the data

        scaled_data = scaler.transform(featrue_test_filled)
        #print(scaled_data)


        pca = PCA(n_components=3)
        principleComp = pca.fit_transform(scaled_data)
        #principleDf = pd.DataFrame(data=principleComp,columns=['principle component 1','principle component 2'])

        sub_arr=[]
        for i in range(len(principleComp)):
            sub_arr.append(int(model.predict([principleComp[i]])[0]))

        submission_copy = csv.reader(open('submission_format.csv'))
        sub_data = list(submission_copy)
        for i in range(len(sub_data)):
            if i == 0:
                continue
            sub_data[i][3] = sub_arr[i-1]
        pd.DataFrame(sub_data).to_csv('SUBMISSION.csv', index=None,header=None)


    def find_incomplete_rows(self):
        empty_cells_map = {}

        array = []
        with open('dengue_features_train.csv', 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            row_counter = 0
            for row in csvreader:
                col_counter = 0
                for col in row:
                    if not col:
                        # check if cur row already has blank
                        if empty_cells_map.get(row_counter, "default value") != "default value":
                            empty_cells_map[row_counter].append(col_counter)
                        else:
                            empty_cells_map[row_counter] = [col_counter]
                        array.append(row_counter)
                    col_counter += 1
                row_counter += 1
        feature_train = pd.read_csv('dengue_features_train.csv')
        rmv_nan_ft = feature_train.dropna()
        label = pd.read_csv("dengue_labels_train.csv",
                         skiprows=array)
        label = label["total_cases"]
        return label
    def pca(self):
        data = pd.read_csv('dengue_features_train.csv.txt')
        scaled_data = preprocessing.scale(data.T)
        pca = PCA()  # create a PCA object
        pca.fit(scaled_data)  # do the math
        pca_data = pca.transform(scaled_data)  # get PCA coordinates for scaled_data


    def model(self,features,label):

        arrayFeature = np.array(features)
        arrayLabel = np.array(label)
        X, y = arrayFeature,arrayLabel

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        model = RandomForestRegressor(random_state=1)
        model.fit(X_train, y_train)
        return model




if __name__ == '__main__':
    a = ProcessData
    #mapEmpty = a.find_incomplete_rows(a)
    pcaFeature = a.cov_matrix(a)
    label = a.find_incomplete_rows(a)
    model = a.model(a,pcaFeature,label)
    a.processTest(a,model)
