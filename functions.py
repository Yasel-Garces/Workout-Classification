#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:34:24 2020
Utilities functions used in the project "Workout Classification"
@author: Yasel Garces (88yasel@gmail.com)
"""
from interval import interval
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
##------------------------------------------------------------------
def iqr_rule(X):
    '''
    Given a serie X, the function the a Serie with the
    same length than X with 1 (inlier) and -1 (outliers)
    '''
    # Compute the .25 and .75 quantile
    quantiles=X.quantile([0.25,0.75])
    # Interquantile range
    IQR=quantiles[0.75]-quantiles[0.25]
    # Create the inlier interval
    inliers_int=interval([quantiles[0.25]-1.5*IQR,quantiles[0.75]+1.5*IQR])
    # label as True the inliers observations and as False the outliers
    result=[x in inliers_int for x in X]
    return result

##------------------------------------------------------------------
def iqr_by_variable(df):
    '''
    Parameters
    ----------
    df : data frame 
        Each columns is a numerical variable.
    Returns
    -------
    list
        1 if all value of the variables are inliers, -1 if not.
    '''
    result=df.apply(iqr_rule,axis=0)
    result=result.sum(axis=1)==df.shape[1]
    return [1 if val else -1 for val in result]
##------------------------------------------------------------------
def find_outliers(df,alt='multv'):
    # Isolation Forest --> Multivariate strategy to find outliers
    # Devices locations
    location=['arm','forearm','belt','dumbbell']
    # Numerical variables to include
    num_variables=['accel_x','accel_y', 'accel_z', 'gyros_x', 'gyros_y','gyros_z', 
               'magnet_x', 'magnet_y', 'magnet_z', 'pitch','roll', 
               'total_accel', 'yaw']
    # Classe
    classe=['A','B','C','D','E']
    # Define expty variables to store the results
    result_by_class={} # Results by each classe
    predic_by_class={} # Individual observation classification by each classe
    for clas in classe:
        # variables to store results
        num_observations=[] # Number of observation by locations
        num_outliers=[] # num of detected outliers
        avg_outliers=[] # Average of the number of outliers
        predictions={} # Predictions for each observation
        for this_location in location:
            # Slice the dataset to consider only a specific location
            temporal=df.loc[(df['Location']==this_location) & 
                                 (df['classe']==clas),num_variables]
            # Check if the alternative is "multv" or univ. If "multv"
            # apply Isolation Forest, else apply Iterquantile Range Rule
            if alt=='multv':
                # Fit the isolation Forest
                clf = IsolationForest(random_state=0).fit(temporal)
                # Make the predictions
                predictions[this_location]=clf.predict(temporal)
            else:
                predictions[this_location]=np.asarray(iqr_by_variable(temporal))
                
            # Update the information
            num_observations.append(len(predictions[this_location]))
            num_outliers.append(sum(predictions[this_location]==-1))
            avg_outliers.append(sum(predictions[this_location]==-1)/len(predictions[this_location]))
        # Save the results of the class in a dictionary 
        result_by_class[clas]=pd.DataFrame({'Location': location,'# Obsv': num_observations,
                 '# Outliers': num_outliers, 'Avg Outliers': avg_outliers})
        predic_by_class[clas]={'Prediction':predictions}
    # Resume the whole information as a data frame
    grl_results=pd.concat({k: pd.DataFrame(v) for k, v in result_by_class.items()}, axis=0)
    grl_results.reset_index(level=0,inplace=True)
    grl_results.rename(columns={'level_0':'Classe'},inplace=True)
    # Include an extra variable with the number of inliers
    grl_results['# Inliers']=grl_results['# Obsv'] - grl_results['# Outliers']
    return (grl_results,predic_by_class)
##------------------------------------------------------------------    