# MyCalibration.py
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
#
import numpy               as np
import matplotlib.pyplot   as plt
import matplotlib.ticker   as ticker
import sklearn.calibration as cal
# import getPDF            as acKDE

def plotCalibrationCurve(plotTitle, dataTitle, params):
    prob_true, prob_predicted = \
        cal.calibration_curve(**params)
        #cal.calibration_curve(labels, scores, normalize=False, strategy=strategy, n_bins=bins)
    actual_bins = len(prob_true)
    bins = params['n_bins']
    if bins > actual_bins:
        print(f'Used {actual_bins} bins instead of the {bins} bins requested.')
    #endif
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(prob_predicted, prob_true, "s-", label=dataTitle)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel('Predicted risk/probability')
    plt.ylabel('Observed risk/probability')
    plt.title(plotTitle)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    ax.axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.axes.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    plotGrey = lambda x, y: plt.fill(x, y, 'k', alpha=0.3, linewidth=None)
    x = []
    y = []
    shadeWidth = int(round(actual_bins / 3))
    step       = shadeWidth * 2     # 3 bins=skip 2;  6 bins=skip 4;  9 bins=skip 6
    for i in range(0, actual_bins, step):
        x0 = i     * 1/actual_bins
        x1 = (i+shadeWidth) * 1/actual_bins
        x = x + [x0] + [x0] + [x1] + [x1]
        y = y +  [0] +  [1] +  [1] +  [0]
    #endfor
    plotGrey(x, y)

    return fig, ax
# enddef

def calibrationOK(numScores, bins):
    if   (numScores/25) >= bins:
        return 1
    elif (numScores/10) >= bins:
        return 0.5
    else:
        return 0
#endif

def doCalibration(scores, labels, posclass, fileNum, showPlot, quiet):

    scores, newlabels, labels = ac.sortScoresFixLabels(scores, labels, posclass, True)  # True = ascending

    maxScore = float(max(scores))
    minScore = float(min(scores))
    if not quiet:
        print(f'Before: min,max = {minScore},{maxScore}')
    # endif
    scores_np = (np.array(scores) - minScore) / (maxScore - minScore)
    maxScore = float(max(scores_np))
    minScore = float(min(scores_np))
    if not quiet:
        print(f'After: min,max = {minScore},{maxScore}')
    # endif
    numScores  = int(len(scores_np))

    #quiet = True
    #Xc_cts, Y = acKDE.getPDF(scores_np, 'epanechnikov', 'new', quiet)
    #Y1D = Y[:, 0]
    #y2 = np.interp(scores_np, Xc_cts, Y1D)
    #plt.plot(scores_np, y2)
    #plt.show()

    for bins in [3, 6, 9]:
        if calibrationOK(numScores, bins) == 0 and (not quiet):
                print(f'Not plotted: insufficient scores ({numScores}) for {bins} bins')
        else:
            plotTitle = f'Calibration plot with {bins} bins'
            dataTitle = 'Classifier'
            if calibrationOK(numScores, bins) == 0.5 and (not quiet):
                print(f'Plotted despite insufficient scores ({numScores}) for {bins} bins')
            # endif
            params = dict(y_true=newlabels, y_prob=scores_np, normalize=False, strategy='uniform', n_bins=bins)
            if showPlot:
                fig, ax = plotCalibrationCurve(plotTitle, dataTitle, quiet, params)
            else:
                prob_predicted, prob_true = dr.computeCalibrationCurve(plotTitle, dataTitle, quiet, params)
            #endif
            if showPlot:
                plt.show()
                fig.savefig(f'output/calib_{fileNum}-{bins}.png')
            #endif
        # endif
    # endfor
#enddef
