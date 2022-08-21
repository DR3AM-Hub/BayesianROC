#!/usr/bin/env python
# BayesianROC.py
# by André Carrington and Franz Mayr

from deeproc.DeepROC import DeepROC

class BayesianROC(DeepROC):

    def __init__(self, predicted_scores=None, labels=None, poslabel=None, BayesianPriorPoint=None,
                 costs=None, quiet=False):
        '''BayesianROC constructor. If predicted_scores and labels are
           empty then it returns an empty object.'''

        super().__init__(predicted_scores=predicted_scores, labels=labels, poslabel=poslabel, quiet=quiet)

        #   Bayesian ROC...
        self.BayesianPriorPoint  = BayesianPriorPoint
        self.costs               = costs
    #enddef

    def analyzeGroupFoldsVsChance(self, groupIndex, prevalence, costs):
        BayesianPriorPoint = (0.5, 0.5)
        forFolds = True
        return self.analyzeGroupVs(groupIndex, prevalence, costs, BayesianPriorPoint, forFolds)
    #enddef

    def analyzeGroupVsChance(self, groupIndex, prevalence, costs):
        BayesianPriorPoint = (0.5, 0.5)
        forFolds = False
        return self.analyzeGroupVs(groupIndex, prevalence, costs, BayesianPriorPoint, forFolds)
    #enddef

    def analyzeGroupFoldsVsPrior(self, groupIndex, prevalence, costs, BayesianPriorPoint):
        if 'BayesianPriorPoint' not in locals():
            BayesianPriorPoint = self.BayesianPriorPoint
        forFolds = True
        return self.analyzeGroupVs(groupIndex, prevalence, costs, BayesianPriorPoint, forFolds)
    # enddef

    def analyzeGroupVsPrior(self, groupIndex, prevalence, costs, BayesianPriorPoint):
        if 'BayesianPriorPoint' not in locals():
            BayesianPriorPoint = self.BayesianPriorPoint
        forFolds = False
        return self.analyzeGroupVs(groupIndex, prevalence, costs, BayesianPriorPoint, forFolds)
    #enddef

    def getMeanROC_A_pi(self, prevalence, costs, BayesianPriorPoint):
        # get the intersection of the prior baseline and the ROC curve, call it A_pi
        from Helpers.BayesianROCFunctions import getA_pi
        if 'BayesianPriorPoint' not in locals():
            BayesianPriorPoint = self.BayesianPriorPoint
        return getA_pi(self.mean_fpr, self.mean_tpr, prevalence, costs, BayesianPriorPoint)
    #enddef

    def getA_pi(self, prevalence, costs, BayesianPriorPoint):
        # get the intersection of the prior baseline and the ROC curve, call it A_pi
        from Helpers.BayesianROCFunctions import getA_pi
        if 'BayesianPriorPoint' not in locals():
            BayesianPriorPoint = self.BayesianPriorPoint
        return getA_pi(self.full_fpr, self.full_tpr, prevalence, costs, BayesianPriorPoint)
    #enddef

    def setBayesianPriorPoint(self, BayesianPriorPoint):
        self.BayesianPriorPoint = BayesianPriorPoint
    #enddef

    def analyzeGroupVs(self, groupIndex, prevalence, costs, BayesianPriorPoint, forFolds):
        from Helpers.BayesianROCFunctions import BayesianAUC

        returnValues     = self.getGroupForAUCi(groupIndex, forFolds)
        groupByOtherAxis = returnValues[3]
        if self.groupAxis == 'FPR':
            group = dict(x1=self.groups[groupIndex][0],
                         x2=self.groups[groupIndex][1],
                         y1=groupByOtherAxis[0],
                         y2=groupByOtherAxis[1])
        elif self.groupAxis == 'TPR':
            group = dict(y1=self.groups[groupIndex][0],
                         y2=self.groups[groupIndex][1],
                         x1=groupByOtherAxis[0],
                         x2=groupByOtherAxis[1])
        else:
            SystemError(f'This function has not been implemented yet for groupAxis=={self.groupAxis}.')
            group = None
        #endif

        if forFolds:
            measures_dict = BayesianAUC(self.mean_fpr, self.mean_tpr, group, prevalence, costs, BayesianPriorPoint)
        else:
            measures_dict = BayesianAUC(self.full_fpr, self.full_tpr, group, prevalence, costs, BayesianPriorPoint)
        #endif

        # # to compute Bhattacharyya Dissimilarity (1 - Bhattacharyya Coefficient) we need to
        # # put the posScores and negScores into bins, and then into dictionaries
        # #    first determine if the distributions are linearly separable, because badly chosen bins
        # #    could hide that. So, we need to choose bins wisely, if they are separable.
        # #
        # # first sort elements, high to low, by score, and within equal scores, by label (in descending order)
        # self.predicted_scores, self.newlabels, self.labels, self.sorted_full_slope_factors = \
        #     sortScoresAndLabels4(self.predicted_scores, self.newlabels, self.labels, self.full_slope_factor)
        # # second, determine seperability
        # # third, get the posScores and negScores so that we can make two binned distributions
        # bDissimilar   = BhattacharyyaDissimilarity()

        return measures_dict
    #enddef

    # modified slightly:
    def plotBayesianIsoLine(self, BayesianPriorPoint, neg, pos, costs):
        from   Helpers.BayesianROCFunctions  import bayesian_iso_lines
        import numpy               as     np
        import matplotlib.pyplot   as     plt

        # plot iso_line that passes through the (bayesian) prior point
        bayes_iso_line_y, bayes_iso_line_x = bayesian_iso_lines(pos/(pos+neg), costs, BayesianPriorPoint)
        x = np.linspace(0, 1, 1000)
        plt.plot(x, bayes_iso_line_y(x), linestyle='-', color='black')
        plt.plot([BayesianPriorPoint[0]], [BayesianPriorPoint[1]], 'ko')
    # enddef

    def plotGroup(self, plotTitle, groupIndex, showError=False, showThresholds=True, showOptimalROCpoints=True,
                  costs=None, saveFileName=None, numShowThresh=20, showPlot=True, labelThresh=True,
                  full_fpr_tpr=True):
        fig, ax = super().plotGroup(plotTitle, groupIndex, showError, showThresholds, showOptimalROCpoints,
                                    costs, saveFileName, numShowThresh, showPlot, labelThresh, full_fpr_tpr)
        pos = 1 / (self.NPclassRatio + 1)
        neg = 1 - pos
        self.plotBayesianIsoLine(self.BayesianPriorPoint, neg, pos, costs)
        return fig, ax
    #enddef

    def plotGroupForFolds(self, plotTitle, groupIndex, foldsNPclassRatio, showError=False, showThresholds=True,
                          showOptimalROCpoints=True, costs=None, saveFileName=None, numShowThresh=20,
                          showPlot=True, labelThresh=True, full_fpr_tpr=True):
        fig, ax = super().plotGroupForFolds(plotTitle, groupIndex, foldsNPclassRatio, showError,
                                            showThresholds, showOptimalROCpoints, costs, saveFileName,
                                            numShowThresh, showPlot, labelThresh, full_fpr_tpr)
        pos = 1/(foldsNPclassRatio + 1)
        neg = 1 - pos
        self.plotBayesianIsoLine(self.BayesianPriorPoint, neg, pos, costs)
        return fig, ax
    #enddef

    def showPointTable(self, groupAxis, groups, fpr, tpr, thresh, A_Omega, indent):
        from Helpers.pointMeasures import optimal_ROC_point_indices
        #from Helpers.pointMeasures import classification_point_measures

        def a(pt, prev):  # accuracy
            return prev * pt[1] + (1-prev) * (1-pt[0])

        def ba(pt):       # balanced accuracy
            return (pt[1] + (1-pt[0]))/2

        def nb(pt, prev, costs):  # avg net benefit
            cFN, cTP, cFP, cTN = costs['cFN'], costs['cTP'], costs['cFP'], costs['cTN']

            fixedCosts = lambda prev, cFN, cTN:      prev  * cFN + \
                                                (1 - prev) * cTN

            return      prev  * (cFN - cTP) * pt[1] - \
                   (1 - prev) * (cFP - cTN) * pt[0] - \
                   fixedCosts(prev, cFN, cTN)
        #enddef

        def cwa(nbValue, minNB):  # cost weighted accuracy
            return (nbValue-minNB)/abs(minNB)

        Omega         = '\u03A9'
        Header1       = ['',            'ROC'  , '',         'Balanced', 'Avg Net', 'Cost Weighted']
        Header2       = ['Description', 'point', 'Accuracy', 'Accuracy', 'Benefit', 'Accuracy'     ]
        for h in [Header1, Header2]:
            print(f'{indent}{h[0]:27s}  {h[1]:14s}  {h[2]:9s}  {h[3]:9s}  {h[4]:9s} {h[5]:14s}')

        rows   = []
        if self.foldsNPclassRatio is not None and self.NPclassRatio is None:
            prev = 1 / (self.foldsNPclassRatio + 1)
        else:
            prev = 1 / (self.NPclassRatio + 1)
        costs  = self.costs
        minNB  = nb((1, 0), prev, costs)

        names  = ['Perfect', 'Perfectly Wrong', 'Binary Chance', 'All Negatives', 'All Positives',
                  f'Intersection point A_{Omega}']
        points = [(0, 1),    (1, 0),             (0.5, 0.5),      (0, 0),          (1, 1), A_Omega]
        for name, pt in zip(names, points):
            nbTemp = nb(pt, prev, costs)
            rows   = rows + [[name, pt, a(pt, prev), ba(pt), nbTemp, cwa(nbTemp, minNB)]]
        #endfor

        # Find which group is the whole, if one is, and print its information
        num_groups  = len(groups)
        whole_group = None
        for g in range(0, num_groups):
            if groups[g] == [0, 1]:  # regardless of float or int type
                #  one of the groups is a whole ROC curve
                whole_group = g
                break  # we only need to find one whole group
            #endif
        #endfor
        if whole_group is not None:
            rows = rows + [[f'Whole ROC (group {g+1}):']]
        else:
            rows = rows + [[f'Whole ROC:']]
        #endif

        # compute slopeOrSkew and newcosts for later use
        # if costs['mode'] == 'individuals':
        if costs['costsAreRates'] == False:
            if self.foldsNPclassRatio is not None and self.NPclassRatio is None:
                slope_factor1 = self.foldsNPclassRatio
            else:
                slope_factor1 = self.NPclassRatio
            slope_factor2 = (costs['cFP'] - costs['cTN']) / (costs['cFN'] - costs['cTP'])
            # slope_factor2 = (costs['FP'] - costs['TN']) / (costs['FN'] - costs['TP'])
            # newcosts      = dict(cFP=costs['FP'], cTN=costs['TN'], cFN=costs['FN'], cTP=costs['TP'])
            # newcosts.update(dict(costsAreRates=False))
        else:
            ValueError('costsAreRates==True not yet supported')
        # endif
        slopeOrSkew = slope_factor1 * slope_factor2

        optROCix = optimal_ROC_point_indices(fpr, tpr, slopeOrSkew)
        pt       = (float(fpr[optROCix[0]]), float(tpr[optROCix[0]]))  # unconstrained opt
        nbTemp = nb(pt, prev, costs)
        rows   = rows + [[f'Unconstrained ROC optimum', pt, a(pt, prev), ba(pt), nbTemp, cwa(nbTemp, minNB)]]

        # pt     = (0.3, 0.3)  # constrained opt
        # nbTemp = nb(pt, prev, costs)
        # rows   = rows + [[f'Constrained ROC optimum', pt, a(pt, prev), ba(pt), nbTemp, cwa(nbTemp, minNB)]]

        # now show info for all other groups
        for g in range(0, num_groups):
            if groups[g] == [0, 1]:  # regardless of float or int type
                continue  # skip any groups that are the whole ROC curve
            rows   = rows + [[f'Group {g+1}:']]

            if (groupAxis == 'FPR' and groups[g][0] == 0) or \
                    (groupAxis == 'TPR' and groups[g][1] == 0):
                rocRuleLeft = 'SW'
                rocRuleRight = 'NE'
            else:
                rocRuleLeft = 'NE'
                rocRuleRight = 'NE'
            # endif
            quiet = True
            pfpr, ptpr, _0, _1, _2, _3, _4 = \
                self.getGroupForAUC(fpr, tpr, thresh, groupAxis, groups[g], rocRuleLeft, rocRuleRight, quiet)
            optROIix = optimal_ROC_point_indices(pfpr, ptpr, slopeOrSkew)
            pt     = (float(pfpr[optROIix[0]]), float(ptpr[optROIix[0]]))  # unconstrained opt
            nbTemp = nb(pt, prev, costs)
            rows   = rows + [[f'Unconstrained ROI_{g+1} optimum', pt, a(pt, prev), ba(pt), nbTemp, cwa(nbTemp, minNB)]]

            # pt     = (0.3, 0.3)  # constrained opt
            # nbTemp = nb(pt, prev, costs)
            # rows   = rows + [[f'Constrained ROI_{g+1} optimum',   pt, a(pt, prev), ba(pt), nbTemp, cwa(nbTemp, minNB)]]
        #endfor

        # format and print all rows
        max_rows = len(rows)
        for y in range(0, max_rows):
            if len(rows[y]) == 1:
                print(f'{indent}{rows[y][0]}')  # subtitle row
            else    :  # content rows
                # show the description and point
                ptText = f'({rows[y][1][0]:0.3f},{rows[y][1][1]:0.3f})'
                print(f'{indent}{rows[y][0]:27s}  {ptText:14s}  ', end='')
                # show the 4 numbers
                for number in range(2, 6):
                    numtext = f'{rows[y][number]:0.4f}'
                    if number == 5:
                        numtext = f'{rows[y][number]*100:0.1f}%'
                    print(f'{numtext:9s}  ', end='')
                #endfor
                print('')
            #endif
        #endfor
    #enddef

    def showAreaTable(self, priorSubscript, groupAxis, groups, listOfGroupMeasures, indent=''):
        import numpy as np

        groupMeasures = listOfGroupMeasures
        Header1       = ['Description', 'AUC', 'AUC or', 'Average', 'Average']
        Header2       = ['', 'part', 'normalized', 'Sens', 'Spec']
        for h in [Header1, Header2]:
            print(f'{indent}{h[0]:22s}  {h[1]:7s}  {h[2]:10s}  {h[3]:7s}  {h[4]:7s}')

        num_groups = len(groups)
        max_rows   = 4 * num_groups
        rows = [None] * max_rows

        for g in range(0, num_groups):
            k = g * 4
            if groups[g] == [0, 1]:  # regardless of float or int type
                # whole curve
                rows[k+0] = [f'group {g + 1}: {groupAxis} {groups[g]}']
                rows[k+1] = ['AUC', '-', groupMeasures[g]['AUC_i'], groupMeasures[g]['pAUC'], groupMeasures[g]['pAUCx']]
                rows[k+2] = ['AUC_d', groupMeasures[g]['AUCi_d'], '-', groupMeasures[g]['pAUC_d'], groupMeasures[g]['pAUCx_d']]
                rows[k+3] = [f'AUC_{priorSubscript}', groupMeasures[g]['AUCi_pi'], '-', groupMeasures[g]['pAUC_pi'], groupMeasures[g]['pAUCx_pi']]
            else:
                rows[k+0] = [f'group {g+1}: {groupAxis} {groups[g]}']
                rows[k+1] = ['AUCni', groupMeasures[g]['AUC_i'], groupMeasures[g]['AUCn_i'], groupMeasures[g]['pAUCn'], groupMeasures[g]['pAUCxn']]
                rows[k+2] = ['AUCni_d', groupMeasures[g]['AUCni_d'], '-', groupMeasures[g]['pAUCn_d'], groupMeasures[g]['pAUCxn_d']]
                rows[k+3] = [f'AUCni_{priorSubscript}', groupMeasures[g]['AUCni_pi'], '-', groupMeasures[g]['pAUCn_pi'], groupMeasures[g]['pAUCxn_pi']]
            # endif
        # endfor

        for y in range(0, max_rows):
            if len(rows[y]) == 1:
                print(f'{indent}{rows[y][0]}')  # subtitle row
            else:  # content rows
                # show the description column
                print(f'{indent}{rows[y][0]:22s}  ', end='')

                # show the 4 numbers
                #   first format the number's decimal aspect
                numtext = [None] * 4
                for number in range(1, 5):
                    if f'{type(rows[y][number])}' == "<class 'str'>":  # e.g., blank or -, instead of number
                        numtext[number-1] = f'{rows[y][number]:s}'
                    else:  # a number
                        numtext[number-1] = f'{rows[y][number]:0.4f}'
                #   then format the text spacing
                print(f'{numtext[0]:7s}  {numtext[1]:10s}  {numtext[2]:7s}  {numtext[3]:7s}')
            # endif
        # endfor
        print('')
    # enddef

    def __str__(self):
        '''This method prints the object as a string of its content re 
           predicted scores, labels, full fpr, full tpr, full thresholds.'''
        super().__str__()

        rocdata = f'BayesianPriorPoint = {self.BayesianPriorPoint}\n'
        return rocdata
    #enddef 