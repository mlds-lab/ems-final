import sys
from sys import argv
import argparse
import os
import pickle
import pandas as pd
# import seaborn; seaborn.set()
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import Imputer
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, accuracy_score
from scipy.stats import pearsonr
from numpy import nanmean
from numpy import nanstd
import json

from sklearn.utils import shuffle
from sklearn.svm import SVR
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import scipy 
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit
import glob

import userid_map
import csv_utility
#from fancyimpute.knn import KNN
#from fancyimpute.soft_impute import SoftImpute
#from fancyimpute.iterative_svd import IterativeSVD
#import fancyimpute
#import fancyimpute.soft_impute
#import fancyimpute.iterative_svd

from pyspark import SparkConf, SparkContext
import socket
import copy

import environment_config as ec

from user_ids import get_ids

ENVIRONMENT = socket.gethostname()

np.random.seed(42)
np.random.seed(42)

sc = ec.get_spark_context(ENVIRONMENT)

def df_shifted(df, target=None, lag=0, f="D"):
    """
    This function lags data features with respect to the target scores.

    Args:
        df (data frame): original data
        target (string): target name
        lag (int): period of the lag
        f (string): Increment to use for time frequency

    Returns:
        lagged data frame (dataframe)

    """

    if not lag and not target:
        return df       
    new = {}
    for c in df.columns:
        if c == target:
            new[c] = df[target]
        else:
            new[c] = df[c].shift(periods=lag, freq=f)
    return pd.DataFrame(data=new)

def corr_score(model,X,y):
    return np.corrcoef(y, model.predict(X))[0,1]

def kcorr_score(model,X,y):
    return scipy.stats.kendalltau(y,model.predict(X))[0]

def raw_df_to_train(df, tr_ids, te_ids, params):
    """
    This function filters the data, imputes missing values, and removes outliers.

    Args:
        df (data frame): input data
        tr_ids (list of ints): indices for the training data
        te_ids (list of ints): indices for the testing data
        params (dictionary): parameters required for filtering/imputation/outlier removal

    Returns:
        numpy arrays for training and testing data features, labels, and user ids.
    """
    
    df2     = copy.deepcopy(df)
    tr_ids2 = tr_ids.copy() 
    te_ids2 = te_ids.copy() 
    params2 = params.copy()
        
    print("Loaded Data Frame with %d instances and %d markers"%(df2.shape[0],df2.shape[1]) )

    df_new = df2[df2.index.get_level_values('Participant').isin(list(tr_ids2)+list(te_ids2) )]
    df=df_new.copy()


    #Fill in the nans on columns that are positive only labels with zeros
    features = list(df.columns)
    zero_fill_columns = []
    for f in features:
        if(".app_usage." in f):
            zero_fill_columns.append(f)
    df_zero_fill = df[zero_fill_columns]
    df_zero_fill=df_zero_fill.fillna(0)
    df[zero_fill_columns] = df_zero_fill
    print(len(zero_fill_columns))

    #Filter users with too few values
    df["Missing Indicator"] = df["org.md2k.data_analysis.feature.phone.driving_total.day"] + \
                                df["org.md2k.data_analysis.feature.phone.bicycle_total.day"] + \
                                df["org.md2k.data_analysis.feature.phone.still_total.day"] +\
                                df["org.md2k.data_analysis.feature.phone.on_foot_total.day"]+ \
                                df["org.md2k.data_analysis.feature.phone.tilting_total.day"]+ \
                                df["org.md2k.data_analysis.feature.phone.walking_total.day"]+\
                                df["org.md2k.data_analysis.feature.phone.running_total.day"]+\
                                df["org.md2k.data_analysis.feature.phone.unknown_total.day"]

    #Collapse the data if intake experiment
    if(params2["experiment-type"]=="intake"):
        df_mean = df.groupby("Participant").mean()
        df_std  = df.groupby("Participant").std()
        df_max  = df.groupby("Participant").max()
        df_min  = df.groupby("Participant").min()
        for c in  df.columns:
            if("target" not in c):
                df_mean[c+"-std"] = df_std[c]
                df_mean[c+"-min"] = 1.0*df_min[c]
                df_mean[c+"-max"] = 1.0*df_max[c]
        df = df_mean
    

    # commenting out to skip case removal for missing data -- ER
    # df.dropna(axis=0, subset=["Missing Indicator"],inplace=True)
    df=df.drop(columns=["Missing Indicator"])

    print("  ... Contains %d instances with core features available"%(df.shape[0]) )

    df.dropna(axis=0, subset=["target"],inplace=True)
    
    print("  ... Contains %d instances with defined targets"%(df.shape[0]) )
        
    df.dropna(axis=1, thresh=params["miss_thresh"]*df.shape[0],inplace=True)
    
    print("  ... Contains %d markers that are >=%.2f pct observed\n"%(df.shape[1], params["miss_thresh"]) )

    # Get original feature, no target
    features = list(df.columns)
    features.remove("target")

    # Add day of week indicators
    if (params2["add_days"]):
        for i in range(7):
            df["day%d"%(i)] = 1*np.array(df.index.get_level_values('Date').map(lambda x: x.dayofweek)==i)

    ###Remove!!!!!
    #df=df.fillna(0)

    # Run Imputation on data frame
    numeric = df[features].as_matrix()
    if(np.any(np.isnan(numeric))):
        
        rank=min(25,max(1,int(0.25*numeric.shape[1])))
        from fancyimpute.iterative_svd import IterativeSVD            
        imp=IterativeSVD(verbose=False, rank=rank,init_fill_method="mean",convergence_threshold=1e-5,random_state=42).complete(numeric)
        #imp=KNN(verbose=True, k=100).complete(numeric)
        
        df[features] = pd.DataFrame(data=imp, columns=features,index=df.index)

    if(params2["add_cum_mean"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cmean"%f]=df[f].groupby("Participant").expanding().mean().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

    if(params2["add_cum_max"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

    if(params2["add_cum_std"]):
        #Add cum-means for all original columns
        for f in features:   
            df["%s-cstd"%f]=df[f].groupby("Participant").expanding().std().values
            #df["%s-cmax"%f]=df[f].groupby("Participant").expanding().max().values

                
    #Lag all columns except for target and day  of week    
    #Add specified lags, but not lag 0
    for l in params2["lags"]:
        if(l>0):
            for f in features:
                df["%s-%d"%(f,l)]=df.groupby(level=0)[f].shift(l)
    #Drop original columns if not using lag 0
    if not (0 in params["lags"]):
        df = df.drop(columns=features)
                
    #Make sure no missing values are left
    df=df.fillna(df.mean())
    
    
    #df_new = copy.deepcopy(df[["target"]])
    if(params2["add_pca"]):
        features = list(df.columns)
        features.remove("target")
        numeric = df[features].as_matrix()
        from sklearn.decomposition import IncrementalPCA
        K= min(params["max_pca_K"],max(1,numeric.shape[1]))
        ipca=IncrementalPCA(n_components=K)
        ipca.fit(numeric)
        Zs = ipca.transform(numeric)
        for k in range (K):
            df["PCA%d"%(k)] = Zs[:,k]


  
    #Sort all columns by name
    cols = list(df.columns)
    cols.sort()
    df = df[cols]
        
    
    df_tr = df[df.index.get_level_values('Participant').isin(list(tr_ids2))]
    df_te = df[df.index.get_level_values('Participant').isin(list(te_ids2))]
    
    #Extract data  matrices
    #Targets
    Y_tr = df_tr["target"].as_matrix().astype(float)
    Y_te = df_te["target"].as_matrix().astype(float)
                                                             
    #Features
    features = list(df.columns)
    features.remove("target")
    X_all = df[features].as_matrix().astype(float)                                                    
    X_tr  = df_tr[features].as_matrix().astype(float)
    X_te  = df_te[features].as_matrix().astype(float)
                                                                                                                 
    #Filter low std columns based on all data
    ind = np.std(X_all,axis=0)>1e-4 
    X_tr = X_tr[:,ind]
    X_te = X_te[:,ind]
    X_all = X_all[:,ind]
    
    #Scale data based on overall mean and std                                                         
    mean = np.mean(X_all,axis=0)
    std  = np.std(X_all,axis=0)                                            
    X_tr = (X_tr-mean)/std
    X_te = (X_te-mean)/std                                                         
                                                             
    features=np.array(features)[ind]
          
          
    if(params2["transfer_filter"]):
        Z = np.hstack((Y_tr[:,np.newaxis],X_tr))
        Corr = np.corrcoef(Z)
        
        X_tr_mean = np.mean(X_tr,axis=0)
        X_tr_std  = np.std(X_tr,axis=0)
        X_te_mean = np.mean(X_te,axis=0)
        X_te_std  = np.std(X_te,axis=0)
        
        Ntr = X_tr.shape[0]
        Nte = X_te.shape[0]
        
        SE = np.sqrt(X_tr_std**2/Ntr + X_te_std**2/Nte)
        tstat = np.abs(X_tr_mean-X_te_mean)/(1e-4+ SE)
        
        ind = tstat <1
        if(np.sum(ind)==0):
            ind[np.argmin(tstat)]=1
        
        X_tr = X_tr[:,ind]
        X_te = X_te[:,ind]
        features=np.array(features)[ind]
        
        print("Filtered %d features"%(np.sum(1-ind)))          
   
    #Row groupings by user id
    G_tr = np.array(df_tr.index.get_level_values(0))
    G_te = np.array(df_te.index.get_level_values(0))
                                                             

    #Quality matrix, full since data already imputed
    Q_tr=1-np.isnan(X_tr)
    Q_te=1-np.isnan(X_te)                                                             

    # Dummy marker groups
    MG=np.arange(X_all.shape[1])
    
    return(X_tr,Y_tr,Q_tr,G_tr,X_te,Y_te,Q_te,G_te,MG, features, df_tr, df_te)

class trainTestPerformanceEstimator:
    def __init__(self, indicator_name, model, features,  hyperparams, cvfolds, cvtype):
        """
        This class estimates the performance of a given estimator model different metrics; In addition, ablation
        testing is performed and a summary of results is produced.

        Args:
            indicator_name (string): score name
            model (object): learning model wrapped in a groupCVLearner's object
            features (list): list of feature names
        """
        
        # self.metrics = [mae, mse, r2_score, lambda x,y:  pearsonr(x,y)[0]]
        # self.metric_names=["MAE", "MSE", "R^2", "r"]
        
        self.metrics = [lambda x,y:  pearsonr(x,y)[0]]
        self.metric_names=["R", "R Best", "NS","ND"]
        # self.metrics = [mae, mse, lambda x,y:  pearsonr(x,y)[0]]
        # self.metric_names=["MAE", "MSE", "R", "NS","ND"]

        self.cvtype=cvtype
        self.cvfolds=cvfolds
        self.hyperparams=hyperparams
        self.model=model
        self.features=features
        self.ablation_scores=None
        self.num_metrics = len(self.metric_names)
        self.indicator_name = indicator_name
        self.bounds = {'stress.d': [1, 5],
                       'anxiety.d': [1, 5],
                       'pos.affect.d': [5, 25],
                       'neg.affect.d': [5, 25],
                       'irb.d': [7, 49],
                       'itp.d': [1, 5],
                       'ocb.d': [0, 8],
                       'cwb.d': [0, 8],
                       'sleep.d': [0, 24],
                       'alc.quantity.d': [0, 20],
                       'tob.quantity.d': [0, 30],
                       'total.pa.d': [0, 8000],
                       'neuroticism.d': [1, 5],
                       'conscientiousness.d': [1, 5],
                       'extraversion.d': [1, 5],
                       'agreeableness.d': [1, 5],
                       'openness.d': [1, 5],
                       'stress': [1, 5],
                       'anxiety': [1, 5],
                       'irb': [7, 49],
                       'itp': [1, 5],
                       'ocb': [20, 100],
                       'inter.deviance': [7, 49],
                       'org.deviance': [12, 84],
                       'shipley.abs': [0, 25],
                       'shipley.vocab': [0, 40],
                       'neuroticism': [1, 5],
                       'conscientiousness': [1, 5],
                       'extraversion': [1, 5],
                       'agreeableness': [1, 5],
                       'openness': [1, 5],
                       'pos.affect': [10, 50],
                       'neg.affect': [10, 50],                       
                       'stai.trait': [20, 80],  
                       'audit': [0, 40], 
                       'gats.status': [1,3], 
                       'gats.quantity': [0, 80], 
                       'ipaq': [0, 35000], 
                       'psqi': [0, 21]
                   }

    def get_indicator_non_outliers(self, y, score_name):
        """
        This method gets the indices of data labels that are not considered as outliers, i.e. data cases with labels
        within the permissible range for the relevant score name.

        Args:
            y (numpy array): labels
            score_name (string): score name

        Returns:
            indices of labels within the permissible range (numpy array of ints)
        """
        
        ind = np.ones(y.shape) > 0

        if score_name in self.bounds.keys():
            score_range = self.bounds[score_name]
        else:
            print("!! Warning -- score name does not exist in bounds list. No outlier removal.")
            return ind
    
        l, h = score_range[0], score_range[1] 
        if(np.isinf(h)):
           h=np.percentile(y, 95)            
            
        ind = np.logical_and(y>=l, y<=h)
    
        return(ind)

    def estimate_performance(self, Xtrain, ytrain, Gtrain, Xtest, ytest, Gtest):
        """
        This method estimates the performance of the trained model via different metrics.

        Args:
            Xtrain (numpy array): training data
            ytrain (numpy array): training labels
            Gtrain (numpy array): training user ids
            Xtest (numpy array): testing data
            ytest (numpy array): testing labels
            Gtest (numpy array): testing user ids
        """
        
        np.random.seed(42)
        np.random.seed(10)
                        
        self.results        = np.zeros((self.num_metrics,2))
        self.opt_params=[]
        
        # Drop y outliers and adjust  y scale to be [0,1]
        # Make sure to scale back when predicting!
        #ind = self.get_indicator_non_outliers(ytrain,self.indicator_name)
        #Xtrain_sub = Xtrain[ind,:]
        #ytrain_sub = ytrain[ind]        

        #Clip targets to range
        Xtrain_sub = Xtrain
        ytrain_sub = ytrain
        ytrain_sub[ytrain_sub<self.bounds[self.indicator_name][0]]=self.bounds[self.indicator_name][0]
        ytrain_sub[ytrain_sub>self.bounds[self.indicator_name][1]]=self.bounds[self.indicator_name][1]


        #scorer = make_scorer(mean_squared_error, greater_is_better=False)
        
        def safe_pearson(x,y):
            c = pearsonr(x,y)[0]
            if(np.isnan(c)): c=0
            return c
                    
        def mae(x,y):
            return np.mean(np.abs(x-y))
            
        def accuracy(x,y):
            return np.sum(x==y)

        if(self.indicator_name=="gats.status"):
            scorer = make_scorer(accuracy)
        else:
            scorer = make_scorer(safe_pearson)

        #scorer = make_scorer(mae, greater_is_better=False)            

        #scorer = make_scorer(r2_score, greater_is_better=True) 

        #Learn the model using grid search CV
        np.random.seed(0)
        model = self.model()

        y_straight_test=[]
        y_straight_train=[]

        if(self.indicator_name=="gats.status"):
            self.best_test_score  = 0
            self.best_train_score = 0
            pass
        else:
            for a in self.hyperparams["alpha_lasso"]:
                for a1 in self.hyperparams["alpha_ridge"]:
                    model.set_params(alpha_lasso=a, alpha_ridge=a1,bounds=self.bounds[self.indicator_name])
                    model.fit(Xtrain_sub, ytrain_sub)
                    y_hat       = model.predict(Xtest)
                    y_hat_train = model.predict(Xtrain)
                    print("a0: %e a1: %e R: %.4f  MSE: %.4f  MAE:%.4f  Lo: %f  Hi %f"%(a,a1,
                        safe_pearson(ytest,y_hat),
                        mean_squared_error(ytest,y_hat),
                          mae(ytest,y_hat),
                          min(y_hat),
                          max(y_hat)))
                    y_straight_test.append(safe_pearson(ytest,y_hat))
                    y_straight_train.append(safe_pearson(ytrain,y_hat_train))
                
            self.best_test_score = np.max(np.array(y_straight_test))
            self.best_train_score = np.max(np.array(y_straight_train))


        np.random.seed(0)
        if(self.cvtype=="shuffle"):
            from sklearn.model_selection import ShuffleSplit
            ss=ShuffleSplit(n_splits=self.cvfolds, random_state=1111, test_size=0.1, train_size=None)
            cv_splits = ss.split(Xtrain_sub)
        elif(self.cvtype=="group"):            
            group_kfold = GroupKFold(n_splits=self.cvfolds)
            cv_splits   = group_kfold.split(Xtrain_sub, ytrain_sub, groups=Gtrain)
        elif(self.cvtype=="groupshuffle"):
            gss = GroupShuffleSplit(n_splits=self.cvfolds, test_size=0.1,  random_state=1234)
            cv_splits   =  gss.split(Xtrain_sub, ytrain_sub, groups=Gtrain)
        elif(self.cvtype=="loo"):
            from sklearn.model_selection import LeaveOneOut
            loo = LeaveOneOut()
            cv_splits  = loo.get_n_splits(Xtrain_sub, ytrain_sub,)
        else:
            print("Error: Cross validation cv_type=%s not specified"%(self.cvtype))
            exit()
        
        this_hyper=self.hyperparams.copy()
        if(self.indicator_name=="gats.status"):
            pass
        else:
            this_hyper["bounds"]=[self.bounds[self.indicator_name]]    
        
        gs          = GridSearchCV(model, this_hyper, scoring=scorer, refit=True, return_train_score=True, cv=cv_splits, verbose=False)
        m           = gs.fit(Xtrain_sub,ytrain_sub)
        cv          = np.vstack((m.cv_results_['mean_test_score'], m.cv_results_['mean_train_score']) )

        self.bounds[self.indicator_name]
        print(self.hyperparams)
        print(cv.T)

        self.trained_model = m.best_estimator_
        self.opt_params = gs.best_params_

        self.yhat_train = m.predict(Xtrain)
        self.yhat_test  = m.predict(Xtest)
        
        print("Pred Extrema: %f %f %f %f"%(min(self.yhat_train),max(self.yhat_train),min(self.yhat_test),max(self.yhat_test)))
        
        self.ytest=ytest
        self.ytrain=ytrain

        self.Gtrain=Gtrain
        self.Gtest=Gtest

        # Ablation test:
        if(self.indicator_name=="gats.status"):
            pass
        else:      
            self.ablation_test(Xtrain_sub, Xtest, ytrain_sub, ytest)
            
        #self.opt_params.append(self.model.opt_params)
        
        
        self.results[0,0] = safe_pearson(ytest,self.yhat_test)
        self.results[0,1] = safe_pearson(ytrain,self.yhat_train)
        self.results[1,0] = self.best_test_score
        self.results[1,1] = self.best_train_score            
        self.results[2,0] = len(np.unique(Gtest))
        self.results[2,1] = len(np.unique(Gtrain))
        self.results[3,0] = len(ytest)
        self.results[3,1] = len(ytrain)

    def ablation_test(self, X_tr, X_te, y_tr, y_te):
        """
        This method performs ablation testing for the model.

        Args:
            X_tr (numpy array): training data features
            X_te (numpy array): testing data features
            y_tr (numpy array): training labels
            y_te (numpy array): testing labels

        Returns:
            ablation scores (numpy array)
        """

        params = self.opt_params
        feature_support = self.trained_model.feature_support

        # handling special cases:
        if len(feature_support) == 0:
            self.ablation_scores = np.array([])
            return self.ablation_scores
        elif len(feature_support) == 1:
            self.ablation_scores = np.array([0])
            return self.ablation_scores

        feature_support_temp = list(feature_support)
        self.ablation_scores = np.zeros((len(feature_support),))

        from MLE.linear_regression_one import LinearRegressionOne

        if isinstance(self.trained_model, LinearRegressionOne):

            from sklearn.linear_model import Ridge
            model = Ridge(alpha=params['alpha_ridge'])

        #elif isinstance(self.trained_model, NNRegressionOne):

        #    from MLE.nn_regression_one import RidgeNet
        #    ridge_layers = copy.copy(params['ridge_layers'])
        #    ridge_layers.insert(0, len(feature_support) - 1)
        #    model = RidgeNet(alpha=params['alpha_ridge'], layers=ridge_layers)

        for i, feature in enumerate(feature_support):

            feature_support_temp.remove(feature)

            model.fit(X_tr[:, feature_support_temp], y_tr.reshape((-1,1)))
            y_predict = model.predict(X_te[:, feature_support_temp]).reshape(-1)
            #yhat_predict = self.clip_prediction(self.Ymean + self.Yscale * y_predict, self.indicator_name)
            self.ablation_scores[i] = pearsonr(y_te, y_predict)[0]

            feature_support_temp.append(feature)

        return self.ablation_scores

    def report(self):
        """
        This method generates a report in table format from the ablation test.

        Returns:
            data frame with features, model weights, and ablation scores as columns
        """

        types  = ["Test","Train"]
        
        dfperf = pd.DataFrame(data=[self.indicator_name], columns=["Indicator"])      
        for i,t in enumerate(types):
            for m in range(len(self.metric_names)):
                dfperf["%s %s"%(t,self.metric_names[m])] = [self.results[m,i]]
        dfperf["Optimal Hyper-Parameters"] = [str(self.opt_params)]

        print('optimal params = {}'.format(self.opt_params))

        try:
            coef = self.trained_model.ridge_coef
            indf = np.argsort(-np.abs(coef))
            L    = len(self.trained_model.feature_support)
            inds = np.argsort(-np.abs(coef[self.trained_model.feature_support]))
            dffeatures = pd.DataFrame(data=list(zip(self.features[indf[:L]], coef[indf[:L]], self.ablation_scores[inds])),
                                     columns = ["Features", "Weight", "Ablation Scores"])
        except:
            dffeatures = pd.DataFrame(columns = ["Features", "Weight", "Ablation Scores"])
            pass

        pd.options.display.width = 300
        pd.options.display.max_colwidth= 300
        
        print(dffeatures)

        return dfperf, dffeatures

def group_train_test_split(X, y, G):
    """
    This function splits the data into training and testing.

    Args:
        X (numpy array): data features
        y (numpy array): labels
        G (numpy array): user ids

    Returns:
        training and testing data/labels/userIDs (numpy arrays)
    """

    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=0)
    ttsplit = gss.split(X, groups=G)
    for train_index, test_index in ttsplit:
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        Gtrain, Gtest = G[train_index], G[test_index]
    return(Xtrain, Xtest, ytrain, ytest, Gtrain, Gtest)

def data_frame_to_csv(df, score_column, score_name, prefix="", results_folder="results/"):
    """
    This function writes the prediction results to csv files.

    Args:
        df (data frame): data cases with prediction results
        score_column (string): name of the column with prediction scores
        score_name (string): score name
        prefix (string): path to write the csv file to and the prefix for file name
    """

    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)

    ids     = np.array(df.index.get_level_values('Participant'))
    umn_ids = userid_map.perform_map(ids, 'data/mperf_ids.txt')
    vals    = np.array(df[score_column].fillna(0))

    dates   = [x.strftime("%-m/%-d/%Y") for x in df.index.get_level_values('Date')]

    csv_utility.write_csv_daily(results_folder + "%s"%(prefix), umn_ids, dates, np.array([""]*len(dates)), score_name, vals)

def short_name_from_pkl(fname):
    """
    Extracts an indicator name from the name of a Qualtrics stream.

    Args:
        fname (str): Name of a file from which to extract the indicator name.

    Returns:
        short_name (str): Name of indicator extracted from fname.
    """

    short_name = fname[len("para_dumpdf_org.md2k.data_qualtrics.feature.v12."):-len(".pkl")]
    return(short_name)

def learn_model_get_results(pkl_dir, pkl_file, edd_directory, edd_name, save=False, results_dir="experiment_output"):
    """
    Primary function responsible for processing incoming data, learning a model and outputting predictions.

    Args:
        pkl_dir (str): Path to the data file for model training.
        pkl_file (str): Name of the master data file for model training.
        exp_parameters (dict): The parameters for model training.
        edd_directory (str): Directory where the specified EDD can be found.
        edd_name (str): Filename of the EDD to read in from edd_directory.
        save (bool): Whether or not to save the results of the model training.
        results_dir (str): Directory into which results should be saved.

    Returns:
        df_perf (pandas DataFrame): DataFrame representing the results of the model training.
    """

    np.random.seed(42)
    np.random.seed(10)

    edd = None
    with open(edd_directory + edd_name) as f:
        edd = json.load(f)

    if edd is not None:
        print("loaded EDD with target: {}".format(edd["target"]["name"]))
    else:
        print("EDD load failed: %s"%(edd_directory + edd_name))
        exit()

    if "exp-parameters" in edd:
        exp_parameters = edd["exp-parameters"]
        
    else:
        print("EDD missing required parameters")
        exit()

    # safety check for existing directory
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # INTERFACE TODO: the name of the experiment will come from the EDD 
    # short_name = short_name_from_pkl(pkl_file) 
    short_name = edd_name[:-len(".json")]


    print("Loaded summary file:",os.path.join(pkl_dir,pkl_file))
    out       = pickle.load( open(os.path.join(pkl_dir,pkl_file), "rb" ) )
    try:    
        df_raw    =  out["dataframe"]
        meta_data = out["metadata"]
    except:
        df_raw = out
    df_raw[df_raw==999] = np.nan

    #Correction to collect gats.status streams
    #Values are converted to strings during CSV generation
    edd_target_stream = edd["target"]["name"]
    # if(edd_target_stream == "org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value"):
    #     level1 = (1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(current)"] ))
    #     level2 = 2*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(past)"]))
    #     level3 = 3*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value(never)"]  ))          
    #     vals = (level1 + level2 + level3) 
    #     valmap=np.array([np.nan,1,2,3])
    #     vals = vals.apply(lambda x: valmap[x])
    #     df_raw["org.md2k.data_qualtrics.feature.v15.igtb.gats.status&value"]=vals

    # updating for the new non-versioned names -- ER
    if(edd_target_stream == "org.md2k.data_qualtrics.feature.igtb.gats.status&value"):
        level1 = (1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.igtb.gats.status&value(current)"] ))
        level2 = 2*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.igtb.gats.status&value(past)"]))
        level3 = 3*(1- np.isnan(df_raw["org.md2k.data_qualtrics.feature.igtb.gats.status&value(never)"]  ))          
        vals = (level1 + level2 + level3) 
        valmap=np.array([np.nan,1,2,3])
        vals = vals.apply(lambda x: valmap[x])
        df_raw["org.md2k.data_qualtrics.feature.igtb.gats.status&value"]=vals
        
    #Deterime what streams are in the EDD    
    edd_marker_streams = ["target"] #add target to avoid dropping later
    for e in edd["marker-streams"]:
        s = e["name"]
        edd_marker_streams.append(s)

    #Check the set of streams in the master
    #This is qualtrics and markers combined
    master_streams = df_raw.columns

    #Copy the target stream to the target field
    
    print(edd_target_stream)
    
    if edd_target_stream in master_streams:
        df_raw["target"] = df_raw[edd_target_stream]
    else:
        print("Warning: The target specified in the EDD does not exist as a stream in the master summary")
        df_empty = pd.DataFrame()
        return(df_empty)

    #Drop all of the columns that are not listed as marker streams
    #in the EDD
    if(exp_parameters["experiment-type"]=="intake"):
        filter_streams = ["org.md2k.data_analysis.feature.phone.driving_total.day",
                            "org.md2k.data_analysis.feature.phone.bicycle_total.day",
                            "org.md2k.data_analysis.feature.phone.still_total.day",
                            "org.md2k.data_analysis.feature.phone.on_foot_total.day",
                            "org.md2k.data_analysis.feature.phone.tilting_total.day",
                            "org.md2k.data_analysis.feature.phone.walking_total.day",
                            "org.md2k.data_analysis.feature.phone.running_total.day",
                            "org.md2k.data_analysis.feature.phone.unknown_total.day"]
        master_streams = df_raw.columns
        cols_to_drop   = list(set(master_streams)-set(filter_streams+edd_marker_streams))
        edd_df_raw     = df_raw.drop(columns = cols_to_drop)
    else:
        # edd_df_raw =df_raw
        filter_streams = ["org.md2k.data_analysis.feature.phone.driving_total.day",
                            "org.md2k.data_analysis.feature.phone.bicycle_total.day",
                            "org.md2k.data_analysis.feature.phone.still_total.day",
                            "org.md2k.data_analysis.feature.phone.on_foot_total.day",
                            "org.md2k.data_analysis.feature.phone.tilting_total.day",
                            "org.md2k.data_analysis.feature.phone.walking_total.day",
                            "org.md2k.data_analysis.feature.phone.running_total.day",
                            "org.md2k.data_analysis.feature.phone.unknown_total.day"]
        master_streams = df_raw.columns
        cols_to_drop   = list(set(master_streams)-set(filter_streams+edd_marker_streams))
        edd_df_raw     = df_raw.drop(columns = cols_to_drop)
    
    #Filter qualtrics and derived streams out    
    master_streams = edd_df_raw.columns
    cols_to_drop=[]
    for s in(master_streams):
        if ("qualtrics" in s) or ("(" in s):
            cols_to_drop.append(s)
    edd_df_raw = edd_df_raw.drop(columns = cols_to_drop)       
    edd_df_streams = edd_df_raw.columns

    
    #Perform a stratified train-test split
    #Based on participant location codes
    all_ids = get_ids(set=exp_parameters["subject_set"])    
    umn_id = userid_map.perform_map(all_ids, "data/mperf_ids.txt")
    location = np.array([int(x[0]) for x in umn_id ]) #Get location inidcator    
    tr_ids,te_ids= train_test_split(all_ids, train_size=exp_parameters["train_test_split"], stratify=location, random_state=11) 

    X_tr, y_tr, Q_tr, G_tr, X_te, y_te, Q_te, G_te, MG, features, df_tr, df_te = raw_df_to_train(edd_df_raw.copy(), tr_ids, te_ids, exp_parameters)

    if exp_parameters["model"] == "lasso-ridge":
        from MLE.linear_regression_one import LinearRegressionOne
        model = LinearRegressionOne
        
    elif exp_parameters["model"] == "nn-regression":
       from MLE.nn_regression_one import NNRegressionOne
       model = NNRegressionOne        
    elif exp_parameters["model"] =="lr":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression
    else:
        raise ValueError('Invalid model!')

    # Estimate performance
    perf=trainTestPerformanceEstimator(short_name,model,features,exp_parameters["hyper_parameters"], exp_parameters["cv_folds"], exp_parameters["cv_type"])
    perf.estimate_performance(X_tr,y_tr,G_tr,X_te,y_te,G_te)
    df_perf,df_features = perf.report()   

    df_te["prediction"] = 0*df_te["target"]
    df_te["prediction"] = perf.yhat_test
    df_tr["prediction"] = 0*df_tr["target"]
    df_tr["prediction"] = perf.yhat_train

    df_te = df_te[["target","prediction"]]
    df_tr = df_tr[["target","prediction"]]

    #df_te_res=df_te["target"]
    #df_te_res["prediction"] = 0*df_te["target"]
    #df_te_res["prediction"] = perf.yhat_test
    
    #df_tr_res=df_tr["target"]
    #df_tr_res["prediction"] = 0*df_tr["target"]
    #df_tr_res["prediction"] = perf.yhat_train    

    exp_fields = ["Indicator","Master Summary"] + list(exp_parameters.keys())
    exp_vals   = [short_name, os.path.join(pkl_dir,pkl_file)] + list(exp_parameters.values())
    df_config = pd.DataFrame(data={"Experiment Parameter": exp_fields, "Value": exp_vals})

    #output={"df_config":df_config, "df_raw":df_raw, "df_tr":df_tr, "df_te":df_te, "df_perf":df_perf,"df_features":df_features,  "perf":perf}
    #output={"df_config":df_config, "df_raw":df_raw, "df_tr":df_tr_res, "df_te":df_te_res, "df_perf":df_perf,"df_features":df_features}

    output={"df_config":df_config, "df_tr":df_tr, "df_te":df_te, "df_perf":df_perf,"df_features":df_features}

    pickle.dump( output, open( results_dir + "%s-%s.pkl"%(exp_parameters["exp_name"],short_name), "wb" ), protocol=2 )

    if(save):
        data_frame_to_csv(df_te, "target", short_name, prefix="ground_truth_")
        data_frame_to_csv(df_te, "prediction", short_name, prefix="prediction_")
        
    return(df_perf)

def parallel_learn_worker(params):
    """
    Parallelizable function responsible for unpacking the experiment parameters pulled from the EDD
    and passing them to learn_model_get_results().

    Args:
        params (dict): The set of all parameters describing the setup for an experiment.

    Returns:
        output of learn_model_get_results() (pandas DataFrame): DataFrame representing the results of the model training.
    """
    
    np.random.seed(42)
    np.random.seed(10)

    #print("incoming params: {}".format(params))

    full_params = json.loads(params).copy()
    pkl_dir = full_params["pkl_dir"]
    pkl_file = full_params["pkl_file"]
    edd_directory = full_params["edd_directory"]
    edd_name = full_params["edd_name"]
    save = full_params["save"]
    results_dir = full_params["results_dir"]

    return learn_model_get_results(pkl_dir, pkl_file, edd_directory, edd_name, save=save, results_dir=results_dir)

def main(args):
    """
    Entry point to the model-training and prediction pipeline following summarization.  
    Responsible for checking and handling run-time arguments,
    packaging up the experiment parameters, setting parallelization (according to the "parallelism" parameter),
    starting execution of the pipeline, and finally collecting, concatenating and display the results of the model
    training.

    Args:
        args (argparse.Namespace): The set of arguments passed in at the command line.
    """
    
    np.random.seed(42)
    np.random.seed(10)

    no_spark = args.no_spark
    
    pd.set_option('display.width', 1000)

    # get data directory, handle if missing
    if args.data_file is None:
        print("Need a path to a master data file")
        exit()


    print("Using master file: %s"%(args.data_file))
    pkl_dir = os.path.dirname(args.data_file)
    pkl_file = os.path.basename(args.data_file)

    if not os.path.isdir(pkl_dir):
        print("data directory not found!")

        # TODO: throw an exception or something to indicate the directory doesn't exist
        os.makedirs(pkl_dir)

    # get EDD directory, handle if missing
    if args.edd_dir is not None:
        edd_directory = args.edd_dir

    if not os.path.isdir(edd_directory):
        print("edd directory not found: {}".format(edd_directory))

        # TODO: again -- handle this better than a silent loop over no content
        os.makedirs(edd_directory)

    # get EDD name, if any
    if args.edd_name is not None:
        edd_name = args.edd_name

    else:
        edd_name = None

    # set output directory
    mdd=os.path.splitext(pkl_file)[0]
    results_dir = "experiment_output/%s/"%(mdd)
    if not os.path.isdir(results_dir):
        try:
            os.makedirs(results_dir)
        except:
            pass

    full_params = {}
    full_params["pkl_dir"] = pkl_dir
    full_params["pkl_file"] = pkl_file
    full_params["edd_directory"] = edd_directory
    full_params["save"] = False
    full_params["results_dir"] = results_dir


    # loop through or parallelize a directory of EDDs
    if edd_name is None:
        all_jobs = []
        for edd_name in os.listdir(edd_directory):
            
            if(True):
            
                if(".json" in edd_name):
                    full_params_copy = full_params.copy()
                    full_params_copy["edd_name"] = edd_name
                    if no_spark:     
                        #try:               
                            df_results = parallel_learn_worker(json.dumps(full_params_copy))
                            print(df_results)
                        #except:
                        #    print("Error: EDD %s could not be processed"%(edd_name))
                            
                        
                    else:
                        # this will be the basis of the new jobs list
                        all_jobs.append(json.dumps(full_params_copy))

        if not no_spark:
            # derive job_list from all_jobs (above)
            job_list = sc.parallelize(all_jobs)
            job_map = job_list.map(parallel_learn_worker)
            df_results_list=job_map.collect()
            df_results = pd.concat(df_results_list)

    # run if EDD is specified at command line
    else:

        print ("edd_name: {}".format(edd_name))
        full_params["edd_name"] = edd_name
        # testing setup prior to parallelizing
        # this_result = parallel_learn_worker(full_params)
        if no_spark:

            df_results = parallel_learn_worker(json.dumps(full_params))
            print(df_results)
        else:
            # testing parallelization
            job_list = sc.parallelize([json.dumps(full_params)])
            job_map = job_list.map(parallel_learn_worker)
            df_results_list=job_map.collect()        
            df_results = pd.concat(df_results_list)
            print(df_results)
    
if __name__ == "__main__":
    """
    Start of execution.  Creates argparse.ArgumentParser() and parses the argument list 
    that main() uses to execute the pipeline.
    """
    
    parser = argparse.ArgumentParser(description="mPerf EMS")
    parser.add_argument("--data-file", help="path to data file")
    parser.add_argument("--edd-dir", help="path to directory containing EDDs")
    parser.add_argument("--edd-name", nargs='?', help="optional: single EDD filename")
    parser.add_argument("--no-spark", action='store_const', const=True, help="parallelism: 'multi' or 'none'")
    
    args = parser.parse_args()

    main(args)















    