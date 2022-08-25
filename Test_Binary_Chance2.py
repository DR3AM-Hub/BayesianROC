#!/usr/bin/env python
# coding: utf-8
# Test_Binary_Chance2.py
# Copyright 2020 Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by Franz Mayr and André Carrington

# Imports
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import random
import time
import warnings
from sklearn.tree               import DecisionTreeClassifier
from sklearn.ensemble           import AdaBoostClassifier
from sklearn.naive_bayes        import MultinomialNB
from sklearn.svm                import LinearSVC
from sklearn.ensemble           import RandomForestClassifier
from sklearn.ensemble           import GradientBoostingClassifier
from sklearn.metrics            import confusion_matrix
from sklearn.metrics            import roc_curve
from sklearn.model_selection    import train_test_split
from scipy.interpolate          import interp1d

from deeproc.Helpers.transcript import start as Transcript_start
from deeproc.Helpers.transcript import stop  as Transcript_stop

from Helpers.bayesianAUC        import getInputsFromUser
from Helpers.acLogging          import findNextFileNumber
from Helpers.rocInputHelp       import getYesAsTrue

# pandas display option: 3 significant digits and all columns
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load and print general parameters
output_dir     = 'output'
fnprefix       = f'{output_dir}/log_chance_'
fnsuffix       = '.txt'
logfn, testNum = findNextFileNumber(fnprefix, fnsuffix)

# capture standard out to logfile
Transcript_start(logfn)

dataWDBC       = getYesAsTrue('Use Wisconsin Diagnostic Breast Cancer? [y]/n', default='y')
if not dataWDBC:
    print(f'Using Statlog Heart.')
#endif

# Inputs
pArea_settings, bAUC_settings, costs, pcosts = getInputsFromUser()
print(f'\npArea_settings: {pArea_settings}')
print(f'bAUC_settings: {bAUC_settings}')
print(f'costs: {costs}')

if dataWDBC:
    # low performance
    # dropSizeTexture = False
    # dropSize        = True
    # dropShape       = True

    # medium performance
    # dropSizeTexture = False
    # dropSize        = False
    # dropShape       = True

    # high performance
    # dropSizeTexture = False
    # dropSize        = False
    # dropShape       = False

    dropSizeTexture = getYesAsTrue('Drop size and texture? y/[n]', default='n')
    dropSize        = getYesAsTrue('Drop size? y/[n]',             default='n')
    dropShape       = getYesAsTrue('Drop shape? [y]/n',            default='y')

    # Load Wisconsin Breast Cancer data and do some data wrangling
    data = pd.read_csv("data.csv")
    print("\nThe data frame has {0[0]} rows and {0[1]} columns.".format(data.shape))

    print('\nRemoving the id column and a hidden last column full of nan.')
    # for some reason the column full of nan does not show in Excel, but is visible when the
    # dataframe is viewed in debug mode in Python/Pycharm
    data.drop(data.columns[[-1]], axis=1, inplace=True)
    data.drop(['id'], axis=1, inplace=True)
    if dropSizeTexture or dropSize:
        c = 0
        if dropSize:
            features = ['radius', 'perimeter', 'area']
            print(f'Dropped the size features ({c} in total).')
        else:
            features = ['radius', 'texture', 'perimeter', 'area']
            print(f'Dropped the size and texture features ({c} in total).')
        #endif
        for feature in features:
            for suffix in ['_mean', '_se', '_worst']:
                data.drop([feature + suffix], axis=1, inplace=True)
                c += 1
            #endfor
        #endfor
    #endif

    if dropShape:
        c = 0
        features = ['smoothness', 'compactness', 'concavity', 'concave points', 'symmetry', 'fractal_dimension']
        for feature in features:
            for suffix in ['_mean', '_se', '_worst']:
                data.drop([feature + suffix], axis=1, inplace=True)
                c += 1
            #endfor
        #endfor
        print(f'Dropped the shape features ({c} in total).')
    #endif

    # data.head(5)

    target       = "diagnosis"
    diag_map     = {'M':1, 'B':0}  # malignant is the positive event, benign is the negative
    data[target] = data[target].map(diag_map) # series.map
    features     = list(data.columns)
    predictors   = features.copy()
    predictors.remove(target)
    random_state = 25
    test_size    = 0.33
else:
    data = pd.read_csv("heart.csv")
    print("\nThe data frame has {0[0]} rows and {0[1]} columns.".format(data.shape))
    target       = "heartDisease"
    diag_map     = {1:0, 2:1}  # 1 absence, 2 presence of heart disease
    data[target] = data[target].map(diag_map) # series.map
    temp         = list(data.columns)
    predictors   = temp.copy()
    predictors.remove(target)
    random_state = 26
    test_size    = 0.20
#endif

# Setup train/test splits and classifiers
X = data[predictors]
y = data[target]

patients = len(y)
pos      = len(y[y == 1])
neg      = len(y[y == 0])
if dataWDBC:
    print(f'\nThe data have {patients} diagnoses, {pos} malignant and {neg} benign.')
else:
    print(f'\nThe data have {patients} diagnoses, {pos} with disease and {neg} without.')
#endif
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
print(f'The test split random seed is {random_state}\n')

def compute_Acc_CostWeightedAcc(thresh, prevalence, newcosts, pred_proba, y_test):
    from Helpers.pointMeasures import classification_point_measures
    pred       = pred_proba.copy()
    pred[pred <  thresh] = 0
    pred[pred >= thresh] = 1
    conf       = confusion_matrix(y_test, pred)
    measure    = classification_point_measures(conf, prevalence, newcosts)
    return measure['Acc'], measure['cwAcc'], measure['fixed_costs'], conf
#enddef

def get_trainer(param, class_weight = None):
        if   param == 'cart_entropy':
            trainer = DecisionTreeClassifier(random_state = 1, criterion = "entropy", class_weight = class_weight)
        elif param == 'cart_gini':
            trainer = DecisionTreeClassifier(random_state = 1, criterion = "gini", class_weight = class_weight)
        elif param == 'svm_linear':
            trainer = LinearSVC()
        elif param == 'ada_boost':
            #Does not support class_weight
            trainer = AdaBoostClassifier(random_state = 1)
        elif param == 'naive_bayes':
            #Does not support random_state or class_weight
            trainer = MultinomialNB()
        elif param == 'rnd_forest':
            trainer = RandomForestClassifier(random_state=1, min_samples_split = 75, class_weight = class_weight)
        elif param == 'gradient_boost':
            #Does not support class_weight
            trainer = GradientBoostingClassifier(random_state = 1)
        #endif
        return trainer
#enddef

def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier
#enddef

def getRange(matchRng, approxRng):
    if matchRng[0] == 'NA':
        a = approxRng[0]
    else:
        a = matchRng[0]
    #endif
    if matchRng[1] == 'NA':
        b = approxRng[1]
    else:
        b = matchRng[1]
    # endif
    return [a, b]
#enddef

def run_classifier(name, X_train, X_test, y_train, y_test, pos, neg, costs):
    from Helpers.pointMeasures import optimal_ROC_point_indices
    from Helpers.pointMeasures import classification_point_measures
    from Helpers.splitData     import getTrainAndValidationFoldData
    from BayesianROC           import BayesianROC

    print("--> start:" + name)
    random.seed(1)
    start       = time.time()

    trainer     = get_trainer(name, class_weight=None)
    trainerCV   = get_trainer(name, class_weight=None)

    splitterType = 'KFold'
    num_folds    = 5
    num_repeats  = 2  # for a Repeated splitterType
    accs         = []
    acc_indices  = []
    total_folds, X_train_df, y_train_s, X_cv_df, y_cv_s = \
        getTrainAndValidationFoldData(X_train, y_train, splitterType, num_folds, num_repeats)

    # compute slopeOrSkew and newcosts for later use
    if costs['mode'] == 'individuals':
        slope_factor1 = neg / pos
        slope_factor2 = (costs['FP'] - costs['TN']) / (costs['FN'] - costs['TP'])
        newcosts      = dict(cFP=costs['FP'], cTN=costs['TN'], cFN=costs['FN'], cTP=costs['TP'])
        newcosts.update(dict(costsAreRates=False))
    else:
        ValueError('mode==rates not yet supported')
    # endif
    slopeOrSkew = slope_factor1 * slope_factor2

    binaryChance   = (0.5, 0.5)
    Omega          = '\u03A9'
    Pi             = '\u03A0'
    priorSubscript = Omega

    ROC = BayesianROC(predicted_scores=None, labels=None, poslabel=None, BayesianPriorPoint=binaryChance,
                      costs=newcosts)
    for f in range(0, total_folds):
        trainerCV.fit(X_train_df[f], y_train_s[f])
        CVproba = trainerCV.predict_proba(X_cv_df[f])
        CVproba = CVproba[:, 1]

        # get the ROC for each fold and store it in the ROC object
        fpr, tpr, threshold = roc_curve(y_cv_s[f], CVproba)
        ROC.set_fold(fpr=fpr, tpr=tpr, threshold=threshold)

        # to get the accuracy for each fold, we measure it at an optimal point
        # so first get the optimal point
        optIndicesROC = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
        opt_threshold   = threshold[optIndicesROC[0]]  # of multiple optima take the first

        # apply the threshold at that optimal point, to obtain a confusion matrix
        pred            = CVproba.copy()
        pred[pred <  opt_threshold] = 0
        pred[pred >= opt_threshold] = 1
        conf            = confusion_matrix(y_cv_s[f], pred)

        # get the measure from the confusion matrix
        prevalence      = pos / (pos + neg)
        measure         = classification_point_measures(conf, prevalence=prevalence, costs=newcosts)
        acc             = measure['Acc']
        accs.append(acc)
        acc_indices.append(optIndicesROC[0])
    #endfor

    # Done gathering each fold

    # Plot the mean ROC with the ROI FPR=[0,0,15] and its areas, highlighted
    # groupAxis           = 'TPR'
    # groups              = [[0.85, 1], [0, 1]]
    groupAxis           = 'FPR'
    groups              = [[0, 0.15], [0, 1]]
    # groups            = [[0, 0.15], [0, 0.023], [0, 1]]
    groupIndex_0_015    = 0
    wholeIndex          = 1
    # groupIndex_0_0023 = 1
    # wholeIndex        = 2

    ROC.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)
    plotTitle  = f'Mean ROC for {name} highlighting group {groupIndex_0_015 + 1}'
    foldsNPclassRatio = neg / pos
    print(f'foldsNPclassratio: {foldsNPclassRatio}')
    ROC.setBayesianPriorPoint(binaryChance)
    ROC.setFoldsNPclassRatio(foldsNPclassRatio)
    fig1, ax1 = ROC.plotGroupForFolds(plotTitle, groupIndex_0_015, foldsNPclassRatio, showError=False,
                                      showThresholds=True, showOptimalROCpoints=True, costs=newcosts,
                                      saveFileName=None, numShowThresh=20, showPlot=False, labelThresh=True,
                                      full_fpr_tpr=True)

    # Measure diagnostic capability in ROI FPR=[0,0.15]
    passed, groupMeasures = ROC.analyzeGroup(groupIndex_0_015, showData=False, forFolds=True, quiet=True)
    # passed, groupMeasures2 = ROC.analyzeGroup(groupIndex_0_0023, showData=False, forFolds=True, quiet=True)
    passed, wholeMeasures = ROC.analyzeGroup(wholeIndex, showData=False, forFolds=True, quiet=True)

    print(f'\nCross-validation results:')

    # getGroups for pAUCxn
    if (groupAxis == 'FPR' and groups[groupIndex_0_015][0] == 0) or \
       (groupAxis == 'TPR' and groups[groupIndex_0_015][1] == 0):
        rocRuleLeft = 'SW'
        rocRuleRight = 'NE'
    else:
        rocRuleLeft = 'NE'
        rocRuleRight = 'NE'
    #endif
    quiet = True
    thresholds = np.ones(ROC.mean_fpr.shape)
    pfpr, ptpr, pthresh, _1, _2, _3, _4 = ROC.getGroupForAUC(ROC.mean_fpr, ROC.mean_tpr, thresholds,
                                                        groupAxis, groups[groupIndex_0_015],
                                                        rocRuleLeft, rocRuleRight, quiet)
    optIndicesROI = optimal_ROC_point_indices(pfpr, ptpr, slopeOrSkew)
    pfpr          = np.array(pfpr)
    ptpr          = np.array(ptpr)
    plt.scatter(pfpr[optIndicesROI], ptpr[optIndicesROI], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                edgecolors='b')

    # Describe absolute and relative performance:
    prevalence = pos / (pos + neg)
    groupMeasures.update(ROC.analyzeGroupFoldsVsChance(groupIndex_0_015, prevalence, newcosts))
    # groupMeasures2.update(ROC.analyzeGroupFoldsVsChance(groupIndex_0_0023, prevalence, newcosts))
    wholeMeasures.update(ROC.analyzeGroupFoldsVsChance(wholeIndex, prevalence, newcosts))
    meanAUC, AUChigh, AUClow, AUCs = ROC.getMeanAUC_andCI()

    CV_acc      = np.mean(accs)
    pAUC        = groupMeasures['pAUC']
    AUCi        = groupMeasures['AUC_i']
    AUCi_d      = groupMeasures['AUCi_d']
    AUCi_b      = groupMeasures['AUCi_pi']

    # get the intersection of a prior baseline and the ROC curve, call it A_pi
    # for the binary chance baseline, call it A_Omega instead
    print(f'\n   Get the intersection of the binary chance baseline and the ROC curve as A_{Omega}')
    A_pi, roc_minus_b = ROC.getMeanROC_A_pi(prevalence, newcosts, binaryChance)
    if A_pi[0] is not None:
        plt.scatter([A_pi[0]], [A_pi[1]], s=40, marker='o', alpha=1, facecolors='w', lw= 2, edgecolors='k')
    #endif
    plotFileName = f'MeanROC_CV_Group{groupIndex_0_015}_{name}_{testNum}'
    fig1.savefig(f'{output_dir}/{plotFileName}.png')

    # show Bayesian ROC analysis on areas
    print('')
    indent = '   '
    ROC.showAreaTable(priorSubscript, groupAxis, groups, [groupMeasures, wholeMeasures], indent)
    ROC.showPointTable(groupAxis, groups, ROC.mean_fpr, ROC.mean_tpr, None, A_pi, indent)

    # show Bayesian ROC analysis on areas

    # show Deep ROC analysis
    # ROC.analyze(forFolds=True)  # cannot use without scores/labels or thresholds

    # Hellinger distance
    # Confirm cost-weighted accuracy at:
    #     intersection of ROC and b_Omega is zero
    fig2, ax2 = ROC.plot_folds(f'Mean ROC for cross-validation with {name}', saveFileName=None, showPlot=False)
    plotFileName = f'MeanROC_CV_{name}_{testNum}'
    fig2.savefig(f'{output_dir}/{plotFileName}.png')

    # fit on whole derivation set, test on test set
    trainer.fit(X_train, y_train)
    pred_proba  = trainer.predict_proba(X_test)
    pred_proba  = pred_proba[:, 1]

    #fpr, tpr, threshold = roc_curve(y_test, pred_proba)
    ROCtest     = BayesianROC(predicted_scores=pred_proba, labels=y_test, poslabel=1, BayesianPriorPoint=(0.5, 0.5),
                              costs=newcosts, quiet=True)
    print(f'\nTest results:')

    # groupAxis = 'FPR'
    # groups    = [[0, 0.15], [0, 0.023], [0, 1]]
    ROCtest.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)
    groupIndex_0_015 = 0
    testAUC     = ROCtest.getAUC()
    fpr, tpr, thresholds  = ROCtest.full_fpr, ROCtest.full_tpr, ROCtest.full_thresholds
    plotTitle   = f'Test ROC for {name} highlighting group {groupIndex_0_015 + 1}'
    ROCtest.setNPclassRatio(foldsNPclassRatio)
    ROCtest.setBayesianPriorPoint(binaryChance)
    fig3, ax3   = ROCtest.plotGroup(plotTitle, groupIndex_0_015, showError=False,
                                    showThresholds=True, showOptimalROCpoints=True, costs=newcosts,
                                    saveFileName=None, numShowThresh=20, showPlot=False, labelThresh=True,
                                    full_fpr_tpr=True)

    optIndicesROC = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
    plt.scatter(fpr[optIndicesROC], tpr[optIndicesROC], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                edgecolors='r')

    A_pi, roc_minus_b = ROCtest.getA_pi(prevalence, newcosts, binaryChance)
    if A_pi[0] is not None:
        plt.scatter([A_pi[0]], [A_pi[1]], s=40, marker='o', alpha=1, facecolors='w', lw=2, edgecolors='k')
    #endif

    pfpr, ptpr, pthresh, _3, _4, matchRng, approxRng = ROCtest.getGroupForAUC(fpr, tpr, thresholds, groupAxis,
                                                                     groups[groupIndex_0_015],
                                                                     rocRuleLeft, rocRuleRight, quiet)
    # rng     = getRange(matchRng, approxRng)
    # pthresh = thresholds[rng[0]:rng[1]+1]
    # if approxRng[0] != 'NA':
    #     pthresh = np.insert(pthresh, 0, pthresh[0])
    # if approxRng[1] != 'NA':
    #     pthresh = np.append(pthresh, pthresh[-1])
    optIndicesROI = optimal_ROC_point_indices(pfpr, ptpr, slopeOrSkew)
    pfpr    = np.array(pfpr)
    ptpr    = np.array(ptpr)
    pthresh = np.array(pthresh)
    plt.scatter(pfpr[optIndicesROI], ptpr[optIndicesROI], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                edgecolors='b')
    plotFileName = f'ROC_Test_{name}_{testNum}'
    fig3.savefig(f'{output_dir}/{plotFileName}.png')

    # use prevalence specified (not specific to the fold or test set)
    passed, groupMeasures = ROCtest.analyzeGroup(groupIndex_0_015, showData=False, forFolds=False, quiet=True)
    passed, wholeMeasures = ROCtest.analyzeGroup(wholeIndex,       showData=False, forFolds=False, quiet=True)
    groupMeasures.update(ROCtest.analyzeGroupVsChance(groupIndex_0_015, prevalence, newcosts))
    wholeMeasures.update(ROCtest.analyzeGroupVsChance(wholeIndex, prevalence, newcosts))

    test_Acc = None
    print('')
    indent = '   '
    ROCtest.showAreaTable(priorSubscript, groupAxis, groups, [groupMeasures, wholeMeasures], indent)
    ROCtest.showPointTable(groupAxis, groups, pfpr, ptpr, pthresh, A_pi, indent)

    # ROCtest.analyze()

    Transcript_stop()
    plt.show()
    Transcript_start(logfn)

    ret = name, CV_acc, meanAUC, pAUC, AUCi, AUCi_d, AUCi_b, test_Acc, testAUC

    end     = time.time()
    elapsed = end - start
    print("--> end:" + name + f" in {elapsed:0.3f} seconds\n")
    return ret
#enddef

# modified a little:
def run_many_classifiers(X_train, X_test, y_train, y_test, pos, neg, costs):
    #classifiers = ['cart', 'svm_linear', 'naive_bayes', 'ada_boost', 'rnd_forest']
    classifiers = ['cart_entropy', 'cart_gini', 'naive_bayes', 'ada_boost', 'rnd_forest']
    # classifiers = ['cart_gini', 'naive_bayes', 'ada_boost', 'rnd_forest']
    results = []

    for name in classifiers:
        result = run_classifier(name, X_train, X_test, y_train, y_test, pos, neg, costs)
        results.append((name, result))
    #endfor
    
    #print(results)
    result_df = pd.DataFrame(columns=['Model', 'CV Acc', 'Mean AUC', 'ROI: pAUC', 'ROI: AUCi',
                                      'ROI: AUCi_d', 'ROI: AUCi_b', 'Test Acc', 'Test AUC'])

    for name, result in results:
        if result is None:
            print('The model failed, so the result is None')
        else:
            name, CV_acc, meanAUC, pAUC, AUCi, AUCi_d, AUCi_b, testAcc, testAUC = result

            # Table of cross-validation and its mean ROC/AUC
            x = result_df.shape[0]
            result_df.loc[x] = [name, CV_acc, meanAUC, pAUC, AUCi, AUCi_d, AUCi_b, testAcc, testAUC]
        #endif
    #endfor

    return tuple(results), result_df
#enddef

result_tuple, result_df = run_many_classifiers(X_train, X_test, y_train, y_test, pos, neg, costs)
result_df.style.format('{:.3f}')
print(result_df)
Transcript_stop()
