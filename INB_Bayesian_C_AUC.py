#!/usr/bin/env python
# coding: utf-8

# # Integrated Net Benefit and Bayesian C and AUC
# Copyright 2020 Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Franz Mayr and André Carrington

# Imports
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
import random
import time
import warnings
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import AdaBoostClassifier
from sklearn.naive_bayes     import MultinomialNB
from sklearn.svm             import LinearSVC
from sklearn.ensemble        import RandomForestClassifier
from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.metrics         import confusion_matrix
from sklearn.metrics         import classification_report
from sklearn.metrics         import roc_curve
from sklearn.model_selection import train_test_split
from   scipy.interpolate     import interp1d
from Helpers import areaMeasures as am, bayesianAUC as ba, pointMeasures as pm
from Helpers import transcript as transcript
from Helpers import acLogging as log

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load and print general parameters
fnprefix       = 'output/log_chance_'
fnsuffix       = '.txt'
logfn, testNum = log.findNextFileNumber(fnprefix, fnsuffix)

# capture standard out to logfile
transcript.start(logfn)

# Inputs
pArea_settings, bAUC_settings, costs, pcosts = ba.getInputsFromUser()
print(f'\npArea_settings: {pArea_settings}')
print(f'bAUC_settings: {bAUC_settings}')
print(f'costs: {costs}')
print(f'pcosts: {pcosts}')

# Load Wisconsin Breast Cancer data and do some data wrangling
data = pd.read_csv("data.csv")
print("\nThe data frame has {0[0]} rows and {0[1]} columns.".format(data.shape))
# # Preview the first 5 lines of the loaded data 
# data.info()
# data.head(5)

print('\nRemoving the last column and the id column')
data.drop(data.columns[[-1]], axis=1, inplace=True)
data.drop(['id'], axis=1, inplace=True)
# data.head(5)

target       = "diagnosis"
diag_map     = {'M':1, 'B':0}  # malignant is the positive event, benign is the negative
data[target] = data[target].map(diag_map) # series.map
features     = list(data.columns)
predictors   = features.copy()
predictors.remove(target)

# Setup train/test splits and classifiers
X = data[predictors]
y = data[target]

patients = len(y)
pos      = len(y[y == 1])
neg      = len(y[y == 0])
print(f'\nThe data have {patients} diagnoses, {pos} malignant and {neg} benign.')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def get_trainer(param, class_weight = None):
        if   param == 'cart_entropy':
            trainer = DecisionTreeClassifier(random_state = 1, criterion = "entropy", class_weight = class_weight)
        elif param == 'cart_gini':
            trainer = DecisionTreeClassifier(random_state = 1, criterion = "gini", class_weight = class_weight)
        elif param == 'svm':
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

# modified slightly:
def plot_bayesian_iso_line(neg, pos, costs):
    #plot iso_line that pass through the bayesian point
    prev       = pos/(neg+pos)     # prevalence
    prior_point= (prev,prev)
    bayes_iso_line_y, bayes_iso_line_x = ba.bayesian_iso_lines(prior_point, neg, pos, costs)
    x          = np.linspace(0, 1, 1000)
    plt.plot(x, bayes_iso_line_y(x), linestyle=':', color = 'green')
    plt.plot([prev], [prev], 'ro')
#enddef

# modified slightly:
def plot_roc(title, fpr, tpr, roc_auc, optimal_score_pt, neg, pos, costs):
    plt.figure()
    linewidth = 2
    #plt.plot([0, 1], [1, 1], color='grey', alpha=0.2, lw=linewidth, linestyle='-')
    plt.plot(fpr, tpr, color='darkorange',
             lw=linewidth, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=linewidth, linestyle='--')
    plt.xlim([-0.01, 1.0])
    #plt.ylim([0.0, 1.05])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.plot([optimal_score_pt[0]], [optimal_score_pt[1]], 'go')
    
    approximation = interp1d(fpr, tpr)
    # suppress the annoying IntegrationWarning
    import warnings
    warnings.filterwarnings('ignore')
    # warnings.simplefilter(action='ignore', category=IntegrationWarning) # this doesn't work

    x             = np.linspace(0, 1, 1000)
    plt.plot(x, approximation(x), linestyle='solid')
    
    plot_bayesian_iso_line(neg, pos, costs)
    
    plt.show()
#enddef

def pretty_print(message):
    print("--> " + message)
#enddef

# modified a little:
def run_classifier(name, X_train, X_test, y_train, y_test, pos, neg, costs):
    from Helpers.pointMeasures import optimal_ROC_point_indices
    from Helpers.splitData import getTrainAndValidationFoldData
    #from DeepROC import DeepROC
    from BayesianROC import BayesianROC

    # ret = None
    # try:
    if True:
        pretty_print("start:" + name)
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
            slope_factor1 = (neg / pos) ** 2
            slope_factor2 = (costs['FPR'] - costs['TNR']) / (costs['FNR'] - costs['TPR'])
            newcosts      = dict(cFP=costs['FPR'], cTN=costs['TNR'], cFN=costs['FNR'], cTP=costs['TPR'])
            newcosts.update(dict(costsAreRates=True))
        # endif
        slopeOrSkew = slope_factor1 * slope_factor2

        binaryChance = (0.5, 0.5)
        ROC = BayesianROC(predicted_scores=None, labels=None, poslabel=None, BayesianPrior=binaryChance,
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
            optimal_indices = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
            opt_threshold   = threshold[optimal_indices[0]]  # of multiple optima take the first

            # apply the threshold at that optimal point, to obtain a confusion matrix
            pred            = CVproba.copy()
            pred[pred <  opt_threshold] = 0
            pred[pred >= opt_threshold] = 1
            conf            = confusion_matrix(y_cv_s[f], pred)

            # get the measure from the confuion matrix
            measure         = pm.classification_point_measures(conf)
            acc             = measure['Acc']
            accs.append(acc)
            acc_indices.append(optimal_indices[0])
        #endfor

        # Done gathering each fold

        # Plot the mean ROC with the ROI FPR=[0,0,15] and its areas, highlighted
        groupAxis  = 'FPR'
        groups     = [[0, 0.15], [0, 0.023], [0, 1]]
        ROC.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)
        groupIndex = 0
        plotTitle  = f'Mean ROC for {name} highlighting group {groupIndex+1}'
        foldsNPclassRatio = neg / pos
        ROC.plotGroupForFolds(plotTitle, groupIndex, foldsNPclassRatio, showError=False, showThresholds=True,
                              showOptimalROCpoints=True, costs=newcosts, saveFileName=None, numShowThresh=20,
                              showPlot=False, labelThresh=True, full_fpr_tpr=True)

        # Measure diagnostic capability in ROI FPR=[0,0.15]
        # as relevant to my use of the SpPin rule and associated concepts
        # 'AUC_i', 'pAUC', 'pAUCx', 'AUCn_i', 'pAUCn', 'pAUCxn'
        ROC.setFoldsNPclassRatio(foldsNPclassRatio)
        passed, groupMeasures = ROC.analyzeGroup(groupIndex, showData=False, forFolds=True, quiet=True)
        groupIndex2 = 1
        passed, groupMeasures2 = ROC.analyzeGroup(groupIndex2, showData=False, forFolds=True, quiet=True)
        wholeIndex = 2
        passed, wholeMeasures = ROC.analyzeGroup(wholeIndex, showData=False, forFolds=True, quiet=True)
        print(f'In the diagnostic ROI {groupAxis}={groups[groupIndex]}')
        #     ROI: pAUCxn (avgSpec) is ideally 95% or higher
        print(f"   Average specificity (pAUCxn) is {groupMeasures['pAUCxn']:0.3f}; ideally 95% or higher")
        #          is the optimal point in the ROI?
        #          get the within-ROI-optimal-point

        # getGroups for pAUCxn
        if (groupAxis == 'FPR' and groups[groupIndex][0] == 0) or \
           (groupAxis == 'TPR' and groups[groupIndex][1] == 0):
            rocRuleLeft = 'SW'
            rocRuleRight = 'NE'
        else:
            rocRuleLeft = 'NE'
            rocRuleRight = 'NE'
        #endif
        quiet = True
        thresholds = np.ones(ROC.mean_fpr.shape)
        pfpr, ptpr, _1, _2, _3, _4 = ROC.getGroupForAUC(ROC.mean_fpr, ROC.mean_tpr, thresholds,
                                                        groupAxis, groups[groupIndex],
                                                        rocRuleLeft, rocRuleRight, quiet)
        optIndicesROI = optimal_ROC_point_indices(pfpr, ptpr, slopeOrSkew)
        pfpr          = np.array(pfpr)
        ptpr          = np.array(ptpr)

        #          maximum PPV (at a point) as a secondary goal
        #          LR+ > 20 is ideal (at a point)
        #          OR > 36 ideally
        #          AUCi   is ideally at least as good as the whole AUC
        #          maximum pAUCn (avgSens) as a secondary goal

        # Describe absolute and relative performance:
        prevalence = pos / (pos + neg)
        groupMeasures.update(ROC.analyzeGroupFoldsVsChance(groupIndex, prevalence, newcosts))
        groupMeasures2.update(ROC.analyzeGroupFoldsVsChance(groupIndex2, prevalence, newcosts))
        wholeMeasures.update(ROC.analyzeGroupFoldsVsChance(wholeIndex, prevalence, newcosts))
        meanAUC, AUChigh, AUClow, AUCs = ROC.getMeanAUC_andCI()
        print(f'Mean AUC is {meanAUC:0.3f} with confidence interval ({AUClow:0.3f}, {AUChigh:0.3f})')
        #     All: AUC_d vs.AUC_Omega of Mean ROC
        print(f"AUC_d        (diagonal)      is {meanAUC-0.5:.3f}")
        print(f"AUC_d        (diagonal)      is {wholeMeasures['AUCi_d']:0.3f}")
        print(f"AUC_b        (binary chance) is {wholeMeasures['AUCi_pi']:0.3f}")

        print(f'In negative ROI {groupAxis}={groups[groupIndex2]}')
        print(f"pAUC_b+      (diagonal)      is {groupMeasures2['pAUC_pi_pos']:0.3f}")
        print(f"pAUC_b-      (diagonal)      is {groupMeasures2['pAUC_pi_neg']:0.3f}")

        print(f'In the diagnostic ROI {groupAxis}={groups[groupIndex]}')
        print(f"AUCn_i                         is {groupMeasures['AUCn_i']:0.3f}")
        #     ROI: AUC_d1 vs.AUC_Omega1
        print(f"AUC_d1       (- diagonal)      is {groupMeasures['AUCi_d']:0.3f}")
        print(f"AUC_b        (- binary chance) is {groupMeasures['AUCi_pi']:0.3f}")
        #          pAUC_d1 vs.pAUC_Omega1
        print(f"pAUC_d1      (- diagonal)      is {groupMeasures['pAUC_d']:0.3f}")
        print(f"pAUC_b       (- binary chance) is {groupMeasures['pAUC_pi']:0.3f}")
        #          pAUCx_d1 vs.pAUCx_Omega1
        print(f"pAUCx_d1     (- diagonal)      is {groupMeasures['pAUCx_d']:0.3f}")
        print(f"pAUCx_b      (- binary chance) is {groupMeasures['pAUCx_pi']:0.3f}")
        #     Note areas of negative utility
        # Hellinger distance
        # Confirm cost-weighted accuracy at:
        #     intersection of ROC and b_Omega is zero
        ROC.plot_folds(f'Mean ROC for cross-validation with {name}', saveFileName=None, showPlot=False)

        # fit on whole derivation set, test on test set
        trainer.fit(X_train, y_train)
        pred_proba  = trainer.predict_proba(X_test)
        pred_proba  = pred_proba[:, 1]

        fpr, tpr, threshold = roc_curve(y_test, pred_proba)
        ROC.set_fpr_tpr(fpr=fpr, tpr=tpr)

        optimal_indices = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
        opt_threshold = threshold[optimal_indices[0]]  # for multiple optimal points, arbitrarily use the first
        best_point    = (fpr[optimal_indices[0]], tpr[optimal_indices[0]])

        plt.scatter(pfpr[optIndicesROI], ptpr[optIndicesROI], s=30, marker='o', alpha=1, facecolors='w',
                    edgecolors='r')
        plt.show()

        pred          = pred_proba.copy()
        pred[pred <  opt_threshold] = 0
        pred[pred >= opt_threshold] = 1
        rep         = classification_report(y_test, pred)
        conf        = confusion_matrix(y_test, pred)
        measure     = pm.classification_point_measures(conf)
        acc         = measure['Acc']

        ret = rep, conf, acc, pred_proba, meanAUC, best_point
    #endif

    # except (KeyboardInterrupt, SystemExit):
    #     raise
    # except Exception:
    #     logging.error(traceback.format_exc())
    # #endtry
    
    end     = time.time()
    elapsed = end - start
    pretty_print("end:" + name + "("+ str(elapsed) +")\n")
    return ret
#enddef

# modified a little:
def run_many_classifiers(X_train, X_test, y_train, y_test, pos, neg, costs):
    #classifiers = ['cart', 'svm', 'naive_bayes', 'ada_boost', 'rnd_forest']
    classifiers = ['cart_entropy', 'cart_gini', 'naive_bayes', 'ada_boost', 'rnd_forest']
    # classifiers = ['cart_gini', 'naive_bayes', 'ada_boost', 'rnd_forest']
    results = []

    for name in classifiers:
        result = run_classifier(name, X_train, X_test, y_train, y_test, pos, neg, costs)
        results.append((name, result))
    #endfor
    
    #print(results)
    #result_df = pd.DataFrame(columns=['model', 'accuracy','cross_val_acc' ,'my_auc','sklearn_auc','bayesian_auc'])
    result_df = pd.DataFrame(columns=['model', 'accuracy', 'cross_val_acc' ,'my_auc','sklearn_auc','bayesian_auc'])

    for name, result in results:
        if result is None:
            print('The model failed, so the result is None')
        else:

            print("")
            print(name,": ")
            rep, conf, acc, y_pred_proba, CV_acc, best_point  = result
            
            a_meas = {'my_auc':None, 'sklearn_auc':None, 'bayesian_auc':None}
            
            roc_data, a_meas = am.roc_data_and_area_measures(y_test, y_pred_proba, best_point, neg, pos,
                                                             pArea_settings, a_meas, costs )
            fpr, tpr, thr    = roc_data
            my_auc           = a_meas['my_auc']
            sklearn_auc      = a_meas['sklearn_auc']  
            bayesian_auc     = a_meas['bayesian_auc']
            print("Accuracy: "         , acc)
            print("My AUC: "           , my_auc)
            print("SKlearn AUC: "      , sklearn_auc)
            print("The Bayesian AUC : ", bayesian_auc)
            title = f'Test ROC for {name}'
            plot_roc(title, fpr, tpr, sklearn_auc, best_point, neg, pos, costs)

            x = result_df.shape[0]
            result_df.loc[x] = [name, acc, CV_acc, my_auc, sklearn_auc, bayesian_auc]
        #endif
    #endfor

    return tuple(results), result_df
#enddef

result_tuple, result_df = run_many_classifiers(X_train, X_test, y_train, y_test, pos, neg, costs)
print(result_df)
