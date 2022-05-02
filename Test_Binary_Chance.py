#!/usr/bin/env python
# coding: utf-8
# Test_Binary_Chance.py
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

# Inputs
pArea_settings, bAUC_settings, costs, pcosts = getInputsFromUser()
print(f'\npArea_settings: {pArea_settings}')
print(f'bAUC_settings: {bAUC_settings}')
print(f'costs: {costs}')
dropSizeTexture = False
dropSize        = False
dropShape       = False

# Load Wisconsin Breast Cancer data and do some data wrangling
data = pd.read_csv("data.csv")
print("\nThe data frame has {0[0]} rows and {0[1]} columns.".format(data.shape))
# # Preview the first 5 lines of the loaded data 
# data.info()
# data.head(5)

print('\nRemoving the last column and the id column')
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

# Setup train/test splits and classifiers
X = data[predictors]
y = data[target]

patients = len(y)
pos      = len(y[y == 1])
neg      = len(y[y == 0])
print(f'\nThe data have {patients} diagnoses, {pos} malignant and {neg} benign.')
random_state = 25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=random_state)
print(f'The test split random seed is {random_state}')

def compute_Acc_CostWeightedAcc(thresh, prevalence, newcosts, pred_proba, y_test):
    from Helpers.pointMeasures import classification_point_measures
    pred       = pred_proba.copy()
    pred[pred <  thresh] = 0
    pred[pred >= thresh] = 1
    conf       = confusion_matrix(y_test, pred)
    print(conf)
    measure    = classification_point_measures(conf, prevalence, newcosts)
    return measure['Acc'], measure['cwAcc'], measure['fixed_costs']
#enddef

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

def plot_bayesian_iso_line(neg, pos, costs):
    # plot iso_line that passes through the bayesian point
    from Helpers.bayesianAUC import bayesian_iso_lines

    prev       = pos/(neg+pos)     # prevalence
    prior_point= (prev,prev)
    bayes_iso_line_y, bayes_iso_line_x = bayesian_iso_lines(prior_point, neg, pos, costs)
    x          = np.linspace(0, 1, 1000)
    plt.plot(x, bayes_iso_line_y(x), linestyle=':', color = 'green')
    plt.plot([prev], [prev], 'ro')
#enddef

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
            prevalence      = pos / (pos + neg)
            measure         = classification_point_measures(conf, prevalence=prevalence, costs=newcosts)
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
        fig1, ax1 = ROC.plotGroupForFolds(plotTitle, groupIndex, foldsNPclassRatio, showError=False,
                                          showThresholds=True, showOptimalROCpoints=True, costs=newcosts,
                                          saveFileName=None, numShowThresh=20, showPlot=False, labelThresh=True,
                                          full_fpr_tpr=True)

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
        plt.scatter(pfpr[optIndicesROI], ptpr[optIndicesROI], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                    edgecolors='b')
        plotFileName = f'MeanROC_CV_Group{groupIndex}_{name}_{testNum}'
        fig1.savefig(f'{output_dir}/{plotFileName}.png')

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

        CV_acc      = np.mean(accs)
        pAUC        = groupMeasures['pAUC']
        AUCi        = groupMeasures['AUC_i']
        AUCi_d      = groupMeasures['AUCi_d']
        AUCi_b      = groupMeasures['AUCi_pi']

        print(f'Mean AUC is {meanAUC:0.3f} with confidence interval ({AUClow:0.3f}, {AUChigh:0.3f})')
        #     All: AUC_d vs.AUC_Omega of Mean ROC
        print(f"Mean AUC - 0.5                 is {meanAUC-0.5:.3f}")
        print(f"AUC_d        (- diagonal)      is {wholeMeasures['AUCi_d']:0.3f}")
        print(f"AUC_b        (- binary chance) is {wholeMeasures['AUCi_pi']:0.3f}")

        if name == 'cart_gini':
            print(f'In the negative ROI {groupAxis}={groups[groupIndex2]} of the Mean ROC plot:')
            print(f"pAUC_b+      (- binary chance) is {groupMeasures2['pAUC_pi_pos']:0.3f}")
            print(f"pAUC_b-      (- binary chance) is {groupMeasures2['pAUC_pi_neg']:0.3f}")
        #endif

        print(f'In the diagnostic ROI {groupAxis}={groups[groupIndex]} of the Mean ROC plot:')
        print(f"AUCn_i                         is {groupMeasures['AUCn_i']:0.3f}")
        print(f"AUCni_d                        is {groupMeasures['AUCni_d']:0.3f}")
        print(f"AUCni_b                        is {groupMeasures['AUCni_pi']:0.3f}")
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
        fig2, ax2 = ROC.plot_folds(f'Mean ROC for cross-validation with {name}', saveFileName=None, showPlot=False)
        plotFileName = f'MeanROC_CV_{name}_{testNum}'
        fig2.savefig(f'{output_dir}/{plotFileName}.png')

        # fit on whole derivation set, test on test set
        trainer.fit(X_train, y_train)
        pred_proba  = trainer.predict_proba(X_test)
        pred_proba  = pred_proba[:, 1]

        #fpr, tpr, threshold = roc_curve(y_test, pred_proba)
        ROCtest    = BayesianROC(predicted_scores=pred_proba, labels=y_test, poslabel=1, BayesianPrior=(0.5, 0.5),
                               costs=newcosts, quiet=True)
        print(f'In the Test ROC plot:')

        # groupAxis  = 'FPR'
        # groups     = [[0, 0.15], [0, 0.023], [0, 1]]
        ROCtest.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)
        groupIndex = 0
        #ROC.set_fpr_tpr(fpr=fpr, tpr=tpr)
        testAUC   = ROCtest.getAUC()
        fpr, tpr, thresholds  = ROCtest.full_fpr, ROCtest.full_tpr, ROCtest.full_thresholds
        # print(f'fpr {fpr}')
        # print(f'tpr {tpr}')
        # print(f'thresholds {thresholds}')
        plotTitle  = f'Test ROC for {name} highlighting group {groupIndex+1}'
        fig3, ax3 = ROCtest.plotGroup(plotTitle, groupIndex, showError=False, showThresholds=True,
                                      showOptimalROCpoints=True, costs=newcosts, saveFileName=None,
                                      numShowThresh=20, showPlot=False, labelThresh=True, full_fpr_tpr=True)

        optimal_indices = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
        plt.scatter(fpr[optimal_indices], tpr[optimal_indices], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                    edgecolors='r')

        pfpr, ptpr, _3, _4, matchRng, approxRng = ROCtest.getGroupForAUC(fpr, tpr, thresholds, groupAxis,
                                                                         groups[groupIndex],
                                                                         rocRuleLeft, rocRuleRight, quiet)
        rng = getRange(matchRng, approxRng)
        # print(f'rng {rng}')
        pthresh = thresholds[rng[0]:rng[1]+1]
        # print(f'pfpr {pfpr}')
        # print(f'ptpr {ptpr}')
        # print(f'pthresh {pthresh}')

        optIndicesROI = optimal_ROC_point_indices(pfpr, ptpr, slopeOrSkew)
        pfpr    = np.array(pfpr)
        ptpr    = np.array(ptpr)
        pthresh = np.array(pthresh)
        plt.scatter(pfpr[optIndicesROI], ptpr[optIndicesROI], s=40, marker='o', alpha=1, facecolors='w', lw=2,
                    edgecolors='b')
        print(f"Optimal ROC points in the ROI of the Test Plot are:")
        z = 0
        for ix in optIndicesROI:
            print(f"  ({pfpr[optIndicesROI[z]]:0.2f}, {ptpr[optIndicesROI[z]]:0.2f})")
            z += 1

        plotFileName = f'ROC_Test_{name}_{testNum}'
        fig3.savefig(f'{output_dir}/{plotFileName}.png')

        # for Test accuracy, choose one of the "overall" optimal thresholds, i.e., it may not be in the ROI
        thresh     = thresholds[optimal_indices[0]]  # arbitrarily choose the first

        test_Acc, test_cwAcc, testFixedCosts = compute_Acc_CostWeightedAcc(thresh, prevalence, newcosts,
                                                                           pred_proba, y_test)
        print(f"For ROC point:  ({fpr[optimal_indices[0]]:0.2f}, {tpr[optimal_indices[0]]:0.2f})")
        print(f'test_Acc {test_Acc:0.2f}, test_cwAcc {test_cwAcc:0.2f}, testFixedCosts {testFixedCosts:0.2f}')

        for namep, t, p in zip(['A-', 'A+', 'ROIopt'],
                               [np.max(thresholds) + 1, np.min(thresholds) - 1, pthresh[optIndicesROI[0]]],
                               [(0, 0), (1, 1), (fpr[optIndicesROI[0]], tpr[optIndicesROI[0]])]):
            xAcc, xcwAcc, xFixedCosts = compute_Acc_CostWeightedAcc(t, prevalence, newcosts, pred_proba, y_test)
            print(f"For ROC point {namep} at ({p[0]:0.2f}, {p[1]:0.2f}), threshold t={t:0.2f}:")
            print(f'Acc {xAcc:0.2f}, cwAcc {xcwAcc:0.2f}, fixedCosts {xFixedCosts:0.2f}')

        for namep, p in zip(['chance', 'perfect'], [(0.5, 0.5), (0, 1)]):
            print(f"For ROC point {namep} at ({p[0]:0.2f}, {p[1]:0.2f})")
            xAcc          = prevalence * p[1] + (1 - prevalence) * p[0]
            xFixedCosts   = prevalence * newcosts['cFN'] + (1 - prevalence) * newcosts['cTN']
            xcwAcc        = prevalence * (newcosts['cFN'] - newcosts['cTP']) * p[1] - \
                            (1 - prevalence) * (newcosts['cFP'] - newcosts['cTN']) * p[0] - \
                            xFixedCosts
            print(f'Acc {xAcc:0.2f}, cwAcc {xcwAcc:0.2f}, fixedCosts {xFixedCosts:0.2f}')

        Transcript_stop()
        plt.show()
        Transcript_start(logfn)
        ret = name, CV_acc, meanAUC, pAUC, AUCi, AUCi_d, AUCi_b, test_Acc, testAUC
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
# header = ['Model', 'CV Accuracy', 'Mean AUC', 'ROI: pAUC', 'ROI: AUCi', 'ROI: AUCi_d', 'ROI: AUCi_b_del',
#           'ROI: AUCni_d', 'ROI:AUCni_b_del']
# for h in header:
#     print(h + '  ', end='')
# print(result_tuple)
result_df.style.format('{:.3f}')
print(result_df)
Transcript_stop()
