import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



"""
learning simple mpdel using perceptrone learning algorithm 
the code is sample from "python-machine-learning-2nd" 
"""

class perceptrone():
    def __init__(self, eta= 0.01, n_iter = 50, random_state=1 ):
        self.eta = eta;
        self.n_iter = n_iter;
        self.random_state = random_state;
        

    def predict(self, X):
        
        net = np.dot(X, self.W_[1:])+ self.W_[0];
        res = np.where(net >= 0, 1, -1);
        
        return res
        
    def fit(self, xi, y):
        #generate random initial state 
        init = np.random.RandomState(1);

        # get initial weights = n_features +1
        # the extra value for general error
        self.W_ = init.normal(loc = 0.0, scale = self.eta, size = xi.shape[1]+1)

        # define error
        self.errors = [];
        
        # loop for every iteration and update the weights every loop
        for _ in range(self.n_iter):
            error = 0;
            for X, Y in zip(xi, y):
                # predicted value is y' or y hat
                yh = self.predict(X);
                update_val = self.eta * (Y-yh);
                
                self.W_[1:] += update_val * X;
                self.W_[0] += update_val;
                # sum the total errors in teh single loop 
                error += int(update_val !=0 );
            self.errors.append(error);
        print("finished successfully")
        return self
    def fix_iris_data(self, data, feature = "length"):
        self.y = df.iloc[0:100, 4].values
        self.y = np.where(self.y == 'Iris-setosa', -1, 1)

        if feature == "length":
            self.X = df.iloc[0:100, [0, 2]].values
        elif feature == "length":
            self.X = df.iloc[0:100, [1, 3]].values
            
        return self.X, self.y

        
    def plot_feature(self):
        y = df.iloc[0:100, 4].values
        y = np.where(y == 'Iris-setosa', -1, 1)
        
        X = df.iloc[0:100, [0, 2]].values
        
        plt.scatter(X[:50, 0], X[:50, 1],color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],color='blue', marker='x', label='versicolor')
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()
        return self

                

    def plot_decision_regions(self, X, y, classifier, resolution=0.02):
         # setup marker generator and color map
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        # plot the decision surface
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # plot class samples
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8,
                c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
               
                
        plt.xlabel('sepal length [cm]')
        plt.ylabel('petal length [cm]')
        plt.legend(loc='upper left')
        plt.show()
        return self





if __name__ ==" __main__":
    
    df = pd.read_csv("<<data directory>>")
    
    model =  perceptrone()
    
    x, y = model.fix_iris_data(df)
    
    ppn = model.fit(x, y)
    
    model.plot_decision_regions( x, y, ppn, resolution=0.02)
    



    
