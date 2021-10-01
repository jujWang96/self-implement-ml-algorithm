from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
class AdaBoost:
    """ AdaBoost enemble classifier from scratch """

    def __init__(self):
        self.stumps = None
        self.stump_weights = None
        self.errors = None
        self.sample_weights = None
        self.classlabel = {1:None,-1:None}
    def _check_X_y(self, X, y):
        """ Validate assumptions about format of input data, convert minor class to 1"""
        assert len(set(y)) == 2, 'Response variable must only have two classes'
        unique, counts = np.unique(y, return_counts=True)
        self.classlabel[1] = unique[counts == min(counts)].item(0)
        self.classlabel[-1] = unique[counts==max(counts)].item(0)
        newy = np.zeros(X.shape[0])
        newy[y==self.classlabel[1] ]=1
        newy[y==self.classlabel[-1] ]=-1
        return X, newy



    def fit(self, X: np.ndarray, y: np.ndarray, iters=50, max_depth = 3, class_c = {1:1,-1:1}):
        """ Fit the model using training data """

        X, y = self._check_X_y(X, y)
        n = X.shape[0]
        # init numpy arrays
        self.sample_weights = np.zeros(shape=(iters, n))
        self.stumps = np.zeros(shape=iters, dtype=object)
        self.stump_weights = np.zeros(shape=iters)
        self.errors = np.zeros(shape=iters)
        self.sample_cost = np.ones(n)/n
        assert set(self.classlabel.values()) == set(class_c.keys()), 'class costs labels must match'

        self.sample_cost[y==1] = class_c[self.classlabel[1]]
        self.sample_cost[y==-1] = class_c[self.classlabel[-1]]

        # initialize weights uniformly
        self.sample_weights[0] = np.ones(shape=n) / n

        for t in range(iters):
            # fit  weak learner
            curr_sample_weights = self.sample_weights[t]
            stump = DecisionTreeClassifier(max_depth=max_depth,random_state=t)#, max_leaf_nodes=2)
            stump = stump.fit(X, y, sample_weight=curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err_w = np.dot( self.sample_cost[(stump_pred != y)],curr_sample_weights[(stump_pred != y)])
            acc_w = np.dot( self.sample_cost[(stump_pred == y)],curr_sample_weights[(stump_pred == y)])
            err = curr_sample_weights[(stump_pred != y)].sum()

            #stump_weight =  np.log((1 + acc_w- err_w) / (1-acc_w+err_w)) / 2
            stump_weight = np.log(acc_w/err_w)/2
            # update sample weights
            new_sample_weights = (
                self.sample_cost*curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            )
            
            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t+1 < iters:
                self.sample_weights[t+1] = new_sample_weights

            # save results of iteration
            self.stumps[t] = stump
            self.stump_weights[t] = stump_weight
            self.errors[t] = err

        return self

    def predict(self, X):
        """ Make predictions using already fitted model """
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])
        output= np.sign(np.dot(self.stump_weights, stump_preds))

        return np.array([self.classlabel[i] for i in output])
