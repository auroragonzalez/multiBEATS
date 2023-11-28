import time
import re
import numpy as np
from utils.new_constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST
from utils.generic_utils import load_dataset_at
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Select dataset
DATASET_INDEX = 3
MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX] 
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX] 
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX] 

X_train, y_train, X_test, y_test, is_timeseries = load_dataset_at(DATASET_INDEX,
                                                                fold_index=None,
                                                                normalize_timeseries=False)

X_train = np.load("/home/.../TimeSeries_Classification/Review/data/AUSLAN_SL4-WD5-nfeat2-red0.316/X_train.npy")
y_train = np.load("/home/.../TimeSeries_Classification/Review/data/AUSLAN_SL4-WD5-nfeat2-red0.316/y_train.npy")
X_test = np.load("/home/.../TimeSeries_Classification/Review/data/AUSLAN_SL4-WD5-nfeat2-red0.316/X_test.npy")
y_test = np.load("/home/.../TimeSeries_Classification/Review/data/AUSLAN_SL4-WD5-nfeat2-red0.316/y_test.npy")

#-------------------------RF GRID-----------------------
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 100, num = 5)]  # Número de trees en cada forest
max_features = ['log2','sqrt',1.0] # Number of features to consider at every split (sqrt, log, None)
max_depth = [int(x) for x in np.linspace(5, 15, num = 2)]  # Maximum number of levels in tree
max_depth.append(None) # Minimum number of samples required to split a node
# min_samples_split = [2,4]  # Minimum number of samples required at each leaf node
# min_samples_leaf = [2,4]   # Method of selecting samples for training each tree
bootstrap = [True]# Create the random grid

rf_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'bootstrap': bootstrap}

params = rf_grid
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = params, cv = 5, verbose = 1)
grid_search.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())


for i in range(3):
    print('Repetición ', str(i))

    start_time = time.time()
    rf = RandomForestClassifier(**grid_search.best_params_)
    rf.fit(X_train.reshape(X_train.shape[0], -1), y_train.ravel())

    accuracy = sum(rf.predict(X_test.reshape(X_test.shape[0], -1))==y_test.ravel())/len(y_test)
    executionTime = (time.time() - start_time)
    params_row = grid_search.best_params_['bootstrap'], grid_search.best_params_['max_depth'], grid_search.best_params_['max_features'], grid_search.best_params_['n_estimators']

    file1 = open("results_rf.txt", "a")  # append mode
    file1.write(re.sub('/', '',str(DATASET_INDEX)) +";" +str(accuracy)+';'+str(executionTime)+';'+str(params_row)+"\n")
    file1.close()