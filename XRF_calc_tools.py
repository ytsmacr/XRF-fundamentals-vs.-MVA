import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib import pyplot as plt
from statistics import median
import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import cross_val_score
from datetime import date
import pickle

fp = "G:\\My Drive\\Darby Work\\XRF fundamentals vs. MVA\\"

# outlier limits (global variable)
limits_path = "Z:\\Millennium Set\\near_zero&outliers_summary_091021.xlsx"
limits = pd.read_excel(limits_path)
limits = pd.Series(limits['Outlier limit'].values, index=limits['Element']).to_dict()

#-------------------------------------------------------------#
#                   Instrument sensitivity                    #
#-------------------------------------------------------------#   

def calculate_sensitivity(filter):
    
    df = pd.read_csv(fp+'misc\\duplicate_spectra_list.csv')
    folder = "Z:\\data_pXRF\\olympus_17_csv\\"
    
    print("Sensitivity calculations for filter", str(filter))
    
    sens_list = []
    err_list = []
    
    # filter by filter type
    temp_df = df[df['filter']==filter]
    
    # get list of samples
    samples = temp_df['sample'].unique()
    
    # for each sample:
    for sample in tqdm(samples):
        
        temp = temp_df[temp_df['sample']==sample]
        
        # get spectra names
        spectra = temp.spectrum.unique()
        if len(spectra) > 2:
            err_list.append(sample)
            continue

        # first spectrum
        spec1 = str(spectra[0])
        temp1 = pd.read_csv(folder+spec1+'.csv', skiprows=20)
        temp1 = temp1.set_index('Channel #')
        temp1 = temp1.iloc[1:2048, :] # remove first and last rows b/c where stuff gets weird
        temp1['Intensity'] = temp1['Intensity'].astype(int)
        
        # second spectrum
        spec2 = str(spectra[1])
        temp2 = pd.read_csv(folder+spec2+'.csv', skiprows=20)
        temp2 = temp2.set_index('Channel #')
        temp2 = temp2.iloc[1:2048, :]
        temp2['Intensity'] = temp2['Intensity'].astype(int)
        
        # merge and calculate sensitivity
        temp = pd.merge(temp1, temp2, left_index=True, right_index=True)
        temp = temp.T
        sens = temp.astype('float').std(axis=0).mean()
        sens_list.append(sens)
        
    print(err_list, "had more than 2 duplicates and were excluded")
    median_sens = round(median(sens_list),9)
    print("Median sensitivity:", round(median_sens,1))
    mean_sens = sum(sens_list) / len(sens_list)
    print("Mean sensitivity:", round(mean_sens,1))
    
    # compare median to mean
    med_c = 'red'
    mean_c = 'blue'

    plt.hist(sens_list, bins=20)
    plt.ylabel("# Standards")
    plt.xlabel("Sensitivity")
    y_bot, y_top = plt.ylim()
    plt.vlines(x=median_sens,
               ymin = 0,
               ymax = y_top,
               colors=med_c,
               label='median')
    plt.vlines(x=mean_sens,
               ymin = 0,
               ymax = y_top,
               colors=mean_c,
               label='mean')
    plt.title("Filter "+str(filter))
    plt.legend()
    plt.ylim((0,y_top))
    plt.show()

    sens_choice = input("Use median or mean sensitivity? ")
    if sens_choice == 'median':
        sensitivity = median_sens
    elif sens_choice == 'mean':
        sensitivity = mean_sens
    else:
        print("Error: input must be either median or mean.")
    
    return sensitivity

#-------------------------------------------------------------#
#   Calculate limits of blank, detection, and quantification  #
#-------------------------------------------------------------#

def calculate_lbdq(element, num_range, sensitivity):

    fp = "G:\\My Drive\\Darby Work\\XRF fundamentals vs. MVA\\python_models\\PLS\\"

    coeff = pd.read_csv(fp+num_range+"\\"+element+"_coeffs.csv")

    # calculate regression vectors
    vector = pow(coeff, 2).sum().pow(.5)  #square root of sum of squares
                                                
    # calculate values
    factors = {
    'LOB' : 1.645,
    'LOD' : 3.3,
    'LOQ' : 10
    }

    lob = factors['LOB'] * sensitivity * vector[0]
    lod = factors['LOD'] * sensitivity * vector[0]
    loq = factors['LOQ'] * sensitivity * vector[0]
        
    return lob, lod, loq

#-------------------------------------------------------------#
#                         PLS Modelling                       #
#-------------------------------------------------------------#

# Make training model

def train_PLS_model(element, f, metadata, spectra, max_components, n_folds, num_range):

    ep = "G:\\My Drive\\Darby Work\\XRF fundamentals vs. MVA\\python_models\\PLS\\"

    # get training data
    if num_range == '0-750':
        train_meta = metadata[
            metadata['Random'] <= 750
        ][['pkey', element]]

    elif num_range == '250-1000':
        train_meta = metadata[
            metadata['Random'] >= 250
        ][['pkey', element]]

    else:
        train_meta = metadata[['pkey', element]]

    # filter for non-outliers
    train_meta = train_meta[
        train_meta[element] <= limits[element]
    ].reset_index(drop=True)

    train_spectra = spectra[train_meta.pkey]
    
    # format training spectra for model
    spec_list = []

    for column in train_spectra.columns:
        spectrum = list(train_spectra[column])
        spec_list.append(spectrum)

    # exit if not enough standards
    if len(spec_list) < 2:
        print("Error: Fewer than two training spectra.")
        return ['NA' * 5]
    
    X_train = np.array(spec_list)
    
    # select relevant metadata
    y_train = train_meta[element].values
    n1 = len(y_train)

    # cross-validation
    cv_dict = {}
    
    for n_components in np.arange(start=2, stop=max_components+1, step=1):
        # define model
        pls = PLSRegression(n_components = n_components, scale=False)
        # run CV and get RMSE
        rmsecv = (-cross_val_score(
            pls, X_train, y_train, cv=n_folds, scoring='neg_root_mean_squared_error'
        )).mean()
        # add results to dictionary
        cv_dict.update({rmsecv : n_components})
    
    # select parameters of model with lowest rmsecv
    best_rmsecv = min(list(cv_dict.keys()))
    best_component = cv_dict[best_rmsecv]
    best_pls = PLSRegression(n_components = best_component, scale=False)
    
    # train model
    best_pls.fit(X_train,y_train)
    # export model
    filename = ep+num_range+"\\"+element+"_"+f+"_model.asc"
    pickle.dump(best_pls, open(filename, 'wb'), protocol=0)
    
    # model coefficients
    coeff = pd.DataFrame(best_pls.coef_)
    # export coeffs
    path = ep+num_range+'\\'+element+'_coeffs.csv'
    coeff.to_csv(path, index=False)
    
    # r-squared
    r2 = best_pls.score(X_train,y_train)
    # predicted training values and RMSE-C
    train_pred = best_pls.predict(X_train)
    train_pred_true = pd.DataFrame({
        'pred' : train_pred.flatten().tolist(),
        'actual' : y_train.flatten().tolist()
    })

    # remove predictions above 100 wt%
    if len(element) > 2:
        train_pred_true = train_pred_true[
            train_pred_true['pred'] <= 100
        ]
    
    # export pred/true
    path = ep+num_range+'\\'+element+"_"+f+'_train_predictions.csv'
    train_pred_true.to_csv(path, index=False)
    
    # get RMSE-C
    rmsec = round(sqrt(mean_squared_error(train_pred_true.pred, train_pred_true.actual)),2)

    return best_rmsecv, best_component, r2, rmsec, n1

#-------------------------------------------------------------#

# Get test data and run predictions

def test_model(element, f, loq, num_range, metadata, spectra):
    
    # select test samples
    if num_range == '0-750':
        test_meta = metadata[
            metadata['Random'] > 750
        ][['pkey', element]]    

    elif num_range == '250-1000':
        test_meta = metadata[
            metadata['Random'] < 250
        ][['pkey', element]]  

    # filter for non-outliers
    test_meta = test_meta[
        test_meta[element] <= limits[element]
    ].reset_index(drop=True)

    test_spectra = spectra[test_meta.pkey]
    
    # exit if not enough standards
    if len(test_meta) < 2:
        print("Error: <2 test standards for", element)
        n2 = 'NA'
        test_r2 = 'NA'
        rmsep = 'NA'
        return n2, test_r2, rmsep
    
    n2 = len(test_meta)

    test_spec_list = []

    for column in test_spectra.columns:
        spectrum = list(test_spectra[column])
        test_spec_list.append(spectrum)

    X_test = np.array(test_spec_list)

    # import model
    fp = "G:\\My Drive\\Darby Work\\XRF fundamentals vs. MVA\\python_models\\PLS\\"
    filename = fp+num_range+"\\"+element+"_"+f+"_model.asc"
    model = pickle.load(open(filename, 'rb'))
    # run predictions
    test_pred = model.predict(X_test)
    # get RMSE-P
    test_pred_true = pd.DataFrame({
        'pred' : test_pred.flatten().tolist(),
        'actual' : test_meta[element]
    })
    # remove predictions below LOQ
    test_pred_true = test_pred_true[
        test_pred_true['pred'] >= loq
    ]
    # remove predictions above 100 wt%
    if len(element) > 2:
        test_pred_true = test_pred_true[
            test_pred_true['pred'] <= 100
        ]
    # export pred/true
    path = fp+num_range+'\\'+element+"_"+f+'_test_predictions.csv'
    test_pred_true.to_csv(path, index=False)
    # get RMSE-P
    rmsep = round(sqrt(mean_squared_error(test_pred_true.pred, test_pred_true.actual)),2)
    # get R2
    r2 = r2_score(test_pred_true.pred,test_pred_true.actual)

    return n2, r2, rmsep

#-------------------------------------------------------------#
