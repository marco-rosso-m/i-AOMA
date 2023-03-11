from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

from rfcore.helpers import *


class RfCore():
    def __init__(self, ICTHRESH, IC, admissiblepar, discardedpar):
        self.Predictionthreshold = ICTHRESH
        self.IC = IC # training target variable of RF need to be booleanize
        self.y = self.IC > self.Predictionthreshold
        self.y = self.y.astype('uint8')
        # training features variables x
        try:
            self.x, self.y  = joint_admissiblepar_discardedpar(admissiblepar, np.stack( discardedpar, axis=0 ), self.y)
        except ValueError as error: 
            # ValueError: need at least one array to stack
            self.x = admissiblepar
    


        self.clf_model = RandomForestClassifier()
        # n_estimators int, default=100 The number of trees in the forest.
        # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
        # max_depthint, default=None The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.

        self.prediction=[]

    def fit(self):
        self.clf_model.fit(self.x, self.y)
    
    def predict(self, sampledpar):
        self.prediction.append( int( self.clf_model.predict(np.array([sampledpar]))[0] ) )
        return self.prediction[-1]
    
    def all_predictions(self):
        return self.prediction
    
    def save_model_to_file(self, RESULTS_PATH):
        with open(RESULTS_PATH + '/random_forest_trained_model.pkl', 'wb') as file_pi:
            pickle.dump(self.clf_model, file_pi)

    def save_prediction_to_file(self, RESULTS_PATH):
        with open(RESULTS_PATH+'/random_forest_predictions.npy', 'wb') as f:
            np.save(f, np.array(self.prediction))

