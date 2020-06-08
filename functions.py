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
import re
##------------------------------------------------------------------

def types_features(df):
    '''
    This function transform the data types
    Input: df - data frame

    Returns
    -------
    data frame with the correct variable types.
    '''
    # Transform to datetime
    date_time_var=['cvtd_timestamp','raw_timestamp_part_1',
               'raw_timestamp_part_2']
    df.loc[:,date_time_var]=df.loc[:,date_time_var].apply(pd.to_datetime)
    # Transform to category
    categories=['user_name','new_window']
    df.loc[:,categories]=df.loc[:,categories].astype('category')
    
    return df
##------------------------------------------------------------------
def create_semi_tidy_data(df,target):
    '''
    Include only one categorical variable with the location 
    of the sensors (belt, arm, forearm, dumbbell) and conserve 
    all the other categories together.

    Parameters
    ----------
    df : data frame
    target: Serie. Target variable

    Returns
    -------
    (df,target) --> tuple with the data frame and the target variable.

    '''
    # Create an index variable to identify uniquely each instance
    df.reset_index(inplace=True)
    # Create a data frame with the categorical variables -------------------
    # Categorical varibles
    cat_var=[string for string in list(df.columns) if 
     re.search(r'^(?!.*(belt|arm|forearm|dumbbell)).', string)]
    # New data frame
    cat_df=df[cat_var]
    #-----------------------------------------------------------------------
    # Create a data frame with the numerical variables and index
    num_var=[string for string in list(df.columns) if 
     re.search(r'belt|forearm|arm|dumbbell|index', string)]
    num_df=df[num_var]
    # Melt the information in num_df
    num_df_melt=pd.melt(num_df,id_vars='index')
    # Split the variables in "variable" and include it in the "num_df_melt" 
    # dataframe
    # This is going to create a new categorical variable called 
    # "Location" (belt, arm, forearm, dumbbell).
    # Also, this code erase the these 4 locations from "variable"
    location_device=pd.Series(num_df_melt['variable']).str.extract(r'(?P<First>\w*)_(?P<Location>forearm|belt|dumbbell|arm)(?P<Second>\w*)')
    num_df_melt['Location']=location_device['Location']
    num_df_melt['variable']=location_device['First'] + location_device['Second']
    # Unmelt to create new variables using the column "variable"
    num_df_melt=num_df_melt.pivot_table(values='value', 
                                        index=['index','Location'],
                                        columns='variable',
                                        aggfunc=lambda x: x)
    num_df_melt.reset_index(inplace=True) # Reset the index
    # Now it's time to merge the dataframes using "index"
    tidy_df=cat_df.merge(num_df_melt,how='left',on='index')
    # Set index
    tidy_df.set_index('index',inplace=True)
    # Convert to category the variable "Location"
    tidy_df['Location']=tidy_df['Location'].astype('category')
    
    # Reshape the target variable
    target = [target[idx] for idx in tidy_df.index]
    
    return (tidy_df,target)
##------------------------------------------------------------------
def preprocess_data(df,columns_out,dictionary,imputer_object):
    '''
    Pre-process the dataset

    Parameters
    ----------
    df : dataframe
    columns_out: List, columns to drop
    imputer_object: object of class SimpleImputer. Impute NaN with the median.
    dictionary: Dictionary with the types of each variable in df

    Returns
    -------
    None.

    '''
    # Transform the data types
    df = types_features(df)
    # Drop the columns with more than 40% of missing data
    df.drop(columns=columns_out,inplace=True)
    # Impute NaN with the median
    df[dictionary['float']]=imputer_object.transform(df[dictionary['float']])
    # Include only one categorical variable with the location of the sensor
    df = create_semi_tidy_data(df)
    
    return df
    
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
    inliers_int=interval([quantiles[0.25]-1.5*IQR, quantiles[0.75]+1.5*IQR])
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