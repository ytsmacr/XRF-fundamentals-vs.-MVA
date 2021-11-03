import pandas as pd
from tqdm.notebook import tqdm
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib import pyplot as plt
from statistics import median
import os

fp = "G:\\My Drive\\Darby Work\\XRF fundamentals vs. MVA\\"


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

def calculate_lbdq(folder, file_list, o1_sens, o2_sens):
    
    coeffs = []
    elem_list = []
    filt_list = []
    lob_list = []
    lod_list = []
    loq_list = []
    
    for filter_n in ['O1', 'O2']:

        sensitivity = o1_sens if filter_n == 'O1' else o2_sens

        # read models
        if filter_n == 'O1': print("LBDQ calculations:")
        
        ftype = str(filter_n) + "_coeff"
        for file in tqdm(file_list):
            if ftype in file:       
                path = folder + file
                data = pd.read_csv(path, skiprows = [0])

                # convert to dataframe
                data = data.T

                # adapt to different element naming b/w datasets
                data.columns = data.iloc[0].map(lambda x: x.split()[0])
                data = data.drop(data.index[0])
                element = data.columns[0]
                
                # populate lists
                elem_list.append(element)
                filt_list.append(filter_n)

                # calculate regression vectors
                vector = pow(data, 2).sum().pow(.5)  #square root of sum of squares
                
                # calculate values
                factors = {
                    'LOB' : 1.645,
                    'LOD' : 3.3,
                    'LOQ' : 10
                }

                lob_list.append(factors['LOB'] * sensitivity * vector[0])
                lod_list.append(factors['LOD'] * sensitivity * vector[0])
                loq_list.append(factors['LOQ'] * sensitivity * vector[0])

    # make dataframe
    df = pd.DataFrame({
        'element' : elem_list,
        'filter' : filt_list,
        'LOB' : lob_list,
        'LOD' : lod_list,
        'LOQ' : loq_list
    })
    
    # change col formats
    cols = df.columns.drop(['element', 'filter'])
    df[cols] = df[cols].apply(pd.to_numeric)
    
    return df

#-------------------------------------------------------------#
#                         PLS Modelling                       #
#-------------------------------------------------------------#

# Get training metadata

def get_train_meta(metadata, element, num_range, filter):

    train_meta = o1_comps if filter == 'O1' else o2_comps
    
    if num_range == '0-750':
        train_meta = train_meta[
            train_meta['Random Number'] <= 750
        ][['pkey', element]]

    elif num_range == '250-1000':
        train_meta = train_meta[
            train_meta['Random Number'] >= 250
        ][['pkey', element]]

    else: train_meta = train_meta[['pkey', element]]

    # filter for non-outliers
    train_meta = train_meta[
        train_meta[element] <= limits[element]
    ].reset_index(drop=True)

    return train_meta

#-------------------------------------------------------------#

# Make training model

def train_PLS_model(element, train_spectra, train_meta, max_components, n_folds):

    ep = fp+"python_models\\PLS\\"
    
    # format training spectra for model
    spec_list = []

    for column in train_spectra.columns:
        spectrum = list(train_spectra[column])
        spec_list.append(spectrum)

    # exit if not enough standards
    if len(spec_list) < 2:
        n_train.append('NaN')
        train_r2s.append('NaN')
        rmsecs.append('NaN')
        return
    
    X_train = np.array(spec_list)
    
    # select relevant metadata
    y_train = train_meta[element].values
    n_train.append(len(y_train))

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
    rmsecvs.append(best_rmsecv)
    best_component = cv_dict[best_rmsecv]
    components.append(best_component)
    best_pls = PLSRegression(n_components = best_component, scale=False)
    
    # train model
    best_pls.fit(X_train,y_train)
    # export model
    filename = ep+num_range+"\\"+element+"_model.asc"
    pickle.dump(best_pls, open(filename, 'wb'), protocol=0)
    
    # model coefficients
    coeff = pd.DataFrame(best_pls.coef_)
    # export coeffs
    path = ep+num_range+'\\'+element+'_coeffs.csv'
    coeff.to_csv(path, index=False)
    
    # r-squared
    r2 = best_pls.score(X_train,y_train)
    train_r2s.append(r2)
    # predicted training values and RMSE-C
    train_pred = best_pls.predict(X_train)
    train_pred_true = pd.DataFrame({
        'pred' : train_pred.flatten().tolist(),
        'actual' : y_train.flatten().tolist()
    })
    # remove predictions below LOQ
    train_pred_true = train_pred_true[
        train_pred_true['pred'] >= loq_key[element]
    ]
    # remove predictions above 100 wt%
    if len(element) > 2:
        train_pred_true = train_pred_true[
            train_pred_true['pred'] <= 100
        ]
    
    # export pred/true
    path = ep+num_range+'\\'+element+'_train_predictions.csv'
    train_pred_true.to_csv(path, index=False)
    
    # get RMSE-C
    rmsec = round(sqrt(mean_squared_error(train_pred_true.pred, train_pred_true.actual)),2)
    rmsecs.append(rmsec)

#-------------------------------------------------------------#

# Get test data and run predictions

def test_model(element, num_range):
    
    # select test samples
    if num_range == 'all':
        rmseps.append('NaN')
        n_test.append('NaN')
        test_r2s.append('NaN')
        return

    elif num_range == '0-750':
        test_meta = metadata[
            metadata['Random Number'] > 750
        ][['pkey', element]]    

    elif num_range == '250-1000':
        test_meta = metadata[
            metadata['Random Number'] < 250
        ][['pkey', element]]  

    # filter for non-outliers
    test_meta = test_meta[
        test_meta[element] <= limits[element]
    ].reset_index(drop=True)
    
    # exit if not enough standards
    if len(test_meta) < 2:
        n_test.append('NaN')
        test_r2s.append('NaN')
        rmseps.append('NaN')
        return
    
    n_test.append(len(test_meta))

    test_spectra = spectra[test_meta.pkey]

    test_spec_list = []

    for column in test_spectra.columns:
        spectrum = list(test_spectra[column])
        test_spec_list.append(spectrum)

    X_test = np.array(test_spec_list)

    # import model
    filename = ep+num_range+"\\"+element+"_model.asc"
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
        test_pred_true['pred'] >= loq_key[element]
    ]
    # remove predictions above 100 wt%
    if len(element) > 2:
        test_pred_true = test_pred_true[
            test_pred_true['pred'] <= 100
        ]
    # export pred/true
    path = ep+num_range+'\\'+element+'_test_predictions.csv'
    test_pred_true.to_csv(path, index=False)
    # get RMSE-P
    rmsep = round(sqrt(mean_squared_error(test_pred_true.pred, test_pred_true.actual)),2)
    rmseps.append(rmsep)
    # get R2
    r2 = r2_score(test_pred_true.pred,test_pred_true.actual)
    test_r2s.append(r2)

#-------------------------------------------------------------#

# Run model functions

def run_full_model(element, num_range, max_components, n_folds):
    elems.append(element)
    ranges.append(num_range)
    # get relevant training data
    train_meta = get_train_meta(element, num_range)
    train_spectra = spectra[train_meta.pkey]
    # train model
    train_model(element, train_spectra, train_meta, max_components, n_folds)
    # optionally test model
    test_model(element, num_range)








#-------------------------------------------------------------#




# Calculate test errors

def calculate_rmsep(comps, folder, file_list, lbdq):
    
    elem_list = []
    filt_list = []
    avg_list = []
    rmsep_list = []
    r2_list = []
    
    for filter_n in ['O1', 'O2']:

        if filter_n == 'O1': print("RMSEP:")
    
        ftype = filter_n + "_test"

        for file in tqdm(file_list):
            if ftype in file:       
                path = (folder + file)
                data = pd.read_csv(path)

                # get element
                element = data.columns[1].split()[0]
                elem_list.append(element)
                filt_list.append(filter_n)

                # format columns
                data.columns = ['pkey', 'Actual', 'Pred']
                data = data.drop([0])
                data.Pred = data.Pred.astype(float)  

                # remove predictions above 100 for majors
                if len(element) > 2:
                    data = data[data.Pred < 100]

                # remove all predictions below 0
                data = data[data.Pred > 0].reset_index(drop=True).sort_index(axis=1)

                # order columns
                data = data[['pkey', 'Actual', 'Pred']].drop_duplicates(subset = 'pkey').sort_values(by='pkey').reset_index(drop=True)

                # subselect relevant reference values
                ref = lbdq[(lbdq['element'].astype(str) == element) &
                           (lbdq['filter'].astype(str) == filter_n)].reset_index(drop=True)

                # add in Actual concentrations
                temp_comps = comps[comps.pkey.isin(data.pkey)].reset_index(drop=True)                   
                data['Actual'] = temp_comps[temp_comps['pkey'] == data['pkey']][element]

                # remove NaN Actual values....which idk why they'd be there
                data = data.dropna()

                # calculate values
                loq = ref['LOQ'].iloc[0]
                # select just predictions above the LOQ
                data = data[data.Pred > loq].reset_index(drop=True)
                # get average concentration
                avg = data['Actual'].mean()
                avg_list.append(avg)
                # get R2
                if len(data) > 1:
                    r2 = r2_score(data.Actual, data.Pred)
                    r2_list.append(r2)
                else: r2_list.append('Not enough test samples above LOQ')
                # get RMSE-P
                data['sqerror'] = (data.Actual - data.Pred).pow(2)
                rmsep = data['sqerror'].mean() ** 0.5
                rmsep_list.append(rmsep)

    
    df = pd.DataFrame({
        "element" : elem_list,
        "filter" : filt_list,
        "avg_comp" : avg_list,
        "RMSEP" : rmsep_list,
        "R2" : r2_list,
    })
    
    
    return df

#-------------------------------------------------------------#

# Accumulate results

def get_results(comps, regression, n_range, o1_sens, o2_sens):
    
    print('Calculating for', regression, n_range)

    folder = fp+"\\models\\"+regression+"\\"+n_range+"\\"
    file_list = os.listdir(folder)
    if len(file_list) == 0:
        print("\tNo files")
        return

    # calculate lbdq
    lbdq = calculate_lbdq(folder, file_list, o1_sens, o2_sens)
    # calculate rmsep with lbdq results
    rmsep = calculate_rmsep(comps, folder, file_list, lbdq)
    # merge results
    df = pd.merge(lbdq, rmsep, how='outer', on=['element', 'filter'])
    df.insert(loc=2, column='num_range', value=n_range)
    df.insert(loc=2, column='regression', value=regression)
    # return full results
    return df 

#-------------------------------------------------------------#
