# TestChanceROC.py
#
# Copyright 2022 Ottawa Hospital and Region Imaging Associates
# Written by André Carrington
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import pandas              as pd
import scipy.io            as sio
from   os.path import splitext
import ntpath
import re
import pickle
from TestVectors import getTestVector

quiet         = False
testNum       = 1
useFile       = True
resultType    = 'matlab'  # matlab or python
matlabResults = ['040', '356', '529', '536', '581', '639', '643']
pythonResults = ['009']
resultIndex   = 5
pythonMinor   = ['nnReLU', 'svcRBF', 'plogr1']
indexMinor    = 0
groupAxis     = 'TPR'
groups        = [[0.0, 0.8], [0.8, 1]]
#groups       = [[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
#groups       = [[0, 1], [0.0, 1/3], [1/3, 2/3], [2/3, 1.0]]
costs         = dict(cFP=1, cFN=4, cTP=0, cTN=0, costsAreRates=False)  # depends on the dataset
popPrevalence = 0.3  # None uses sample prevalence. population depends on the dataset,
                     # 0.297 for LBC data in 536 and 639, 0.345 for WBC data.

def testChanceROC(descr, scores, labels, groupAxis='FPR', groups=[[0.0, 1/3], [1/3, 2/3], [2/3, 1.0]],
                  costs=dict(cFP=1, cFN=1, cTP=0, cTN=0, costsAreRates=False), popPrevalence=None,
                  showPlot = True):
    from BayesianROC                  import BayesianROC
    from ConcordanceMatrixPlot        import ConcordanceMatrixPlot
    from Helpers.BayesianROCFunctions import showBayesianAUCmeasures
    from Helpers.BayesianROCFunctions import resolvePrevalence
    from Helpers.DeepROCFunctions     import areEpsilonEqual
    from Helpers.DeepROCFunctions     import printGroups

    poslabel = 1
    numShowThresh = 20

    ######################
    print('\naChanceROC:')
    chancePrior = (0.5, 0.5)
    aChanceROC  = BayesianROC(predicted_scores=scores, labels=labels, poslabel=poslabel,
                              BayesianPrior=chancePrior, costs=costs, quiet=quiet)

    if showPlot:
        print('\nROC plot: ', end='')
        print('plot shown.')
        aChanceROC.plot(plotTitle=f'Full ROC Plot for Test {descr}',
                        showThresholds=True, showOptimalROCpoints=True, costs=costs,
                        saveFileName=None, numShowThresh=numShowThresh, showPlot=showPlot)
    #endif

    if showPlot:
        print('\nConcordanceMatrixPlot(aChanceROC):')
        aCMplot = ConcordanceMatrixPlot(aChanceROC)
        print('Plot shown.')
        aCMplot.plot(plotTitle=f'Concordance Matrix for Test {descr}',
                     showThresholds=True, showOptimalROCpoints=True, costs=costs,
                     saveFileName=None, numShowThresholds=numShowThresh, showPlot=showPlot, labelThresh=True)
    #endif

    print(f'\naChanceROC.setGroupsBy: {groupAxis} = ', end='')
    printGroups(groups)
    aChanceROC.setGroupsBy(groupAxis=groupAxis, groups=groups, groupByClosestInstance=False)
    numgroups = len(groups)

    # used in several sections that follow plotGroup, analyzeGroupVsChance
    prevalence = resolvePrevalence(popPrevalence, aChanceROC.newlabels)
    aChanceROC.prevalenceToPlot = prevalence
    aChanceROC.priorToPlot      = chancePrior

    if showPlot:
        print('\naChanceROC.plotGroup:')
        aCMplot2 = ConcordanceMatrixPlot(aChanceROC)
        for i in range(0, numgroups):
            print(f'ROC plot shown for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
            aChanceROC.plotGroup(plotTitle=f'Deep ROC Plot for group {i+1}, Test {descr}', groupIndex=i,
                                 showError=True, showThresholds=True, showOptimalROCpoints=True, costs=costs,
                                 saveFileName=None, numShowThresh=numShowThresh, showPlot=True, labelThresh=True,
                                 full_fpr_tpr=True)
            # the following code commented out has bugs...
            # print(f'ConcordanceMatrixPlot shown for group {i} [{groups[i][0]:0.2f}, {groups[i][1]:0.2f}]')
            # aCMplot2.plotGroup(plotTitle=f'Concordance Matrix Plot for group {i+1}, Test {descr}',
            #                    groupIndex=i, showThresholds=True, showOptimalROCpoints=True, costs=costs,
            #                    saveFileName=None, numShowThresholds=numShowThresh, showPlot=showPlot, labelThresh=True)
    #endfor

    print('\naChanceROC.analyzeGroup:')
    aChanceROC.analyze()

    print('\naChanceROC.analyzeGroupVsChance:')
    cpAUCi_pi_sum = 0
    for i in range(0, numgroups):
        measures_dict = aChanceROC.analyzeGroupVsChance(i, prevalence, costs)
        if i == 0:
            print('Overall across all groups:')
            AUC_pi = measures_dict['AUC_pi']  # we also use this variable later
            print(f"AUC_pi{' ':10s} =  {AUC_pi:0.4f}  (normalized de facto)\n")
        #endif
        print(f'Analyzing group {i+1}:')
        showBayesianAUCmeasures(i+1, measures_dict, aChanceROC.groupsArePerfectCoveringSet)
        if aChanceROC.groupsArePerfectCoveringSet:
            cpAUCi_pi_sum = cpAUCi_pi_sum + measures_dict['cpAUCi_pi']
        #endif
    #endfor
    if aChanceROC.groupsArePerfectCoveringSet:
        #ep        = 1 * (10 ** -12)
        ep         = 1 * (10 ** -3)
        quietFalse = False
        print('Note: Normally we check equality to the 12th decimal place (below), but the equality below')
        print('      is consistent only to the 3rd decimal, for any set of costs and any data. Perhaps the')
        print('      slight imprecision which only occurs sometimes, comes from the interpolated approximation')
        print('      functions specific to this part in our code--to be determined.')
        areEpsilonEqual(cpAUCi_pi_sum, AUC_pi, 'cpAUCi_pi_sum', 'AUC_pi', ep, quietFalse)
    #endif
#enddef

def myLoadFile(resultNumString):
    fileName       = f'input-matlab/result{resultNumString}.mat'  # a matlab file (or future: csv file) for input
    scoreVariable  = 'yscoreTest'                  # yscoreTest, yscore
    targetVariable = 'ytest'                       # ytest, yhatTest, yhat, ytrain
    return loadMatlabOrCsvFile(fileName, scoreVariable, targetVariable)
#enddef

def loadMatlabOrCsvFile(fileName, scoreVariable, targetVariable):
    if fileName == '':
        SystemError('fileName required.')
    else:  # reduce filename to any 3-digit log number it contains, if possible
        fileNameBase = ntpath.basename(fileName)
        fileNameBase = splitext(fileNameBase)[0]  # remove the extension
        match = re.search(r'\d\d\d', fileNameBase)
        if match:
            fileNum = match.group()
        else:
            fileNum = fileNameBase
        # endif
    #endif

    if fileName[-4:] == '.mat':  # if matlab file input
        try:
            fileContent = sio.loadmat(fileName)  # handle any file not found errors naturally
            scores = fileContent[scoreVariable]
            labels = fileContent[targetVariable]
        except:
            raise ValueError(f'File {fileName} is either not found or is not a matlab file')
        # endtry
    else:  # otherwise assume a CSV file input
        try:
            file_df = pd.read_csv(fileName)
            scores  = file_df[scoreVariable]
            labels  = file_df[targetVariable]
        except:
            raise ValueError(f'File {fileName} is either not found or is not a CSV file')
        #endtry
    #endif

    return scores, labels
#enddef


#######  START OF MAIN LOGIC
if useFile:
    if resultType == 'matlab':
        resultNumString = matlabResults[resultIndex]
        print(f'Test Result {resultNumString}')
        fileName        = f'input-matlab/result{resultNumString}.mat'
        scoreVariable   = 'yscoreTest'  # yscoreTest, yscore
        targetVariable  = 'ytest'       # ytest, yhatTest, yhat, ytrain
        #targetVariable  = 'yhatTest'       # ytest, yhatTest, yhat, ytrain
        scores, labels  = loadMatlabOrCsvFile(fileName, scoreVariable, targetVariable)
        scores          = scores.flatten()
        labels          = labels.flatten()

    elif resultType == 'python':
        resultNumString    = f'{pythonResults[resultIndex]}_{pythonMinor[indexMinor]}'
        print(f'Test Result {resultNumString}')
        fileName           = f'input-python/labelScore_{resultNumString}.pkl'
        dataFile           = open(fileName, 'rb')
        [tlabels, tscores] = pickle.load(dataFile)
        #labels             = tlabels[0].flatten()
        labels             = tlabels[1].tolist()
        scores             = tscores[1].flatten().tolist()
        #labels             = tla[]
        #for t in templabels:
        #    labels = labels + [t[1]]
        dataFile.close()

    else:
        ValueError('resultType not recognized')
        resultNumString, scores, labels = None, None, None  # eliminates static warnings
    #endif
else:
    resultNumString = f'{testNum}'
    print(f'Test Vector {resultNumString}')
    scores, labels, dummy_groups, dummy_groupAxis, descr = getTestVector(testNum, noError=False)
#endif

testChanceROC(resultNumString, scores, labels, groupAxis=groupAxis, groups=groups, costs=costs,
              popPrevalence=popPrevalence, showPlot=True)