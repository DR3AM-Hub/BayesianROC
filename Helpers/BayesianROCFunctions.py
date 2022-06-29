# BayesianAUC.py
# Copyright 2020 André Carrington, Ottawa Hospital Research Institute
# Use is subject to the Apache 2.0 License
# Written by André Carrington
#
# main functions:
#   getInputsFromUser
#   makeTieList
#   bayesian_iso_line
#   classifier_area_measures_and_plot_roc
#   PeirceMetzCost
#   CarringtonCost
#   bayesian_auc

import warnings

from   scipy.interpolate     import interp1d
import scipy.integrate       as     integrate

def PeirceMetzCost(prevalence, costs):
    # costs for individuals -- the approach/paradigm found in the literature
    # Peirce 1884, Metz 1978
    if costs['costsAreRates']==True:
        SystemError('The PeiceMetzCost is for costs which are not rates.')
    else:
        pos    = prevalence
        neg    = 1 - prevalence
        cost_P = costs['cFN']  - costs['cTP']
        cost_N = costs['cFP']  - costs['cTN']
        m      = (neg/pos)   * (cost_N/cost_P)
        return m
   #endif
#enddef

def CarringtonCost(prevalence, costs):
    # costs for rates is an alternative paradigm which has a squared effect
    # Carrington 2020
    if costs['costsAreRates']==False:
        SystemError('CarringtonCost is for costs which are rates.')
    else:
        pos    = prevalence
        neg    = 1 - prevalence
        cost_P = costs['cFN']   - costs['cTP']
        cost_N = costs['cFP']   - costs['cTN']
        m      = ((neg/pos)**2) * (cost_N/cost_P)
    return m
    #endif
#enddef

def bayesian_iso_lines(prevalence, costs, prior):
    # iso performance line that passes through a point representing a Bayesian prior (e.g., prevalence)
    x_prior = prior[0]
    y_prior = prior[1]
    
    if costs['costsAreRates']:
        m = CarringtonCost(prevalence, costs)
    else:
        m = PeirceMetzCost(prevalence, costs)
    #endif

    def line_y(x):
        import numpy as np
        # this is a vectorized function
        # force y to stay within the ROC plot by top-coding    y to 1 with np.minimum
        # force y to stay within the ROC plot by bottom-coding y to 0 with np.maximum
        y = m*(x-x_prior)+y_prior
        if isinstance(y, np.ndarray):
            y = np.minimum(y, np.ones(y.shape))
            y = np.maximum(y, np.zeros(y.shape))
        else:
            y = min(y, 1)
            y = max(y, 0)
        #endif
        return y
    #enddef

    def line_x(y):
        import numpy as np
        # this is a vectorized function
        # force x to stay within the ROC plot by top-coding    x to 1 with np.minimum
        # force x to stay within the ROC plot by bottom-coding x to 0 with np.maximum
        x = (y-y_prior)/m + x_prior
        if isinstance(x, np.ndarray):
            x = np.minimum(x, np.ones(x.shape))
            x = np.maximum(x, np.zeros(x.shape))
        else:
            x = min(x, 1)
            x = max(x, 0)
        #endif
        return x
    # enddef

    return line_y, line_x  # return a function of x, not a value
#enddef

def resolvePrevalence(popPrevalence, newlabels):
    if popPrevalence == None:
        # if population prevalence not specified, then use the test sample prevalence
        P = sum(newlabels(newlabels == 1))
        N = sum(newlabels(newlabels == 0))
        prevalence = P / (P+N)
        print(f'Using the sample prevalence of {prevalence} because the population prevalence was not specified.')
    else:
        # population prevalence is specified
        prevalence = popPrevalence
        print(f'Using the population prevalence of {prevalence} as specified.')
    #endif
    return prevalence
#enddef

def resolvePrior(BayesianPrior, prevalence):
    if BayesianPrior == None:
        # BayesianPrior not specified then use the prevalence as the prior (population or sample, as above)
        prior = (prevalence, prevalence)
        print(f'Using the prevalence {prior} as the prior, because the Bayesian prior was not specified.')
    else:
        prior = BayesianPrior
        print(f'Using the Bayesian prior {prior}, as specified.')
    #endif
#enddef

def plot_major_diagonal():
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.linspace(0, 1, 3)
    plt.plot(x, x, linestyle='--', color='black')  # default linewidth is 1.5
    plt.plot(x, x, linestyle='-', color='black', linewidth=0.25)
#enddef

def plot_bayesian_iso_line(prevalence, costs, prior):
    import matplotlib.pyplot as plt
    import numpy as np
    bayes_iso_line_y, bayes_iso_line_x = bayesian_iso_lines(prevalence, costs, prior)
    x = np.linspace(0, 1, 1000)
    plt.plot(x, bayes_iso_line_y(x), linestyle='-', color='black')
    plt.plot(prior[0], prior[1], 'ro')
#enddef

def ChanceAUC(fpr, tpr, group, prevalence, costs):
    prior = (0.5, 0.5)  # binary chance
    return BayesianAUC(fpr, tpr, group, prevalence, costs, prior)
#enddef

def alterForRoot(fpr, tpr):
    import numpy as np

    fpr = list(fpr)
    tpr = list(tpr)
    n = len(fpr)
    for i in range(n-1, -1, -1):
        if tpr[i] == 1:
            continue
        else:
            removeIndex = i+1
            break
        #endif
    #endfor
    if removeIndex < n-1:
        fpr[removeIndex:n-1]=[]  # remove repeated values of [x,1] and just leave the last
        tpr[removeIndex:n-1]=[]  # remove repeated values of [x,1] and just leave the last
    #endif
    n = len(fpr)
    for i in range(0, n):
        if fpr[i] == 0:
            continue
        else:
            removeIndex = i-1
            break
        #endif
    #endfor
    if removeIndex > 0:
        fpr[1:removeIndex+1] = []  # remove repeated values of [x,1] and just leave the last
        tpr[1:removeIndex+1] = []  # remove repeated values of [x,1] and just leave the last
    # endif
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    return fpr, tpr
#enddef

def getA_pi(fpr, tpr, prevalence, costs, prior):
    #from scipy.optimize import root as findRoot
    from scipy.optimize import brentq

    # find the intersection A_pi as the zeros or roots of the equation for
    # the ROC curve (approximate_y, or approximate_x) minus the bayesian_iso_line (b_iso_line_y, or b_iso_line_x)
    # where minus is vertical or horizontal respectively.

    # repeated (0,y) at the beginning of an ROC curve, and repeated (x,1) at the end
    # causes the ROC curve not to be a proper vertical function nor proper horizontal function, respectively
    # and this means that interp1d cannot help us with values or intersections in those two ranges
    # which are not proper functions. Hence, we use a close approximation that is a proper function:
    # we eliminate the repeated points that make it improper, and this is close enough, unless the data
    # are very coarse/sparse.
    fprForRoot, tprForRoot     = alterForRoot(fpr, tpr)

    approximate_y              = interp1d(fprForRoot, tprForRoot)  # x,y; to estimate y=f(x)
    approximate_x              = interp1d(tprForRoot, fprForRoot)  # y,x; to estimate x=f_inverse(y)
    b_iso_line_y, b_iso_line_x = bayesian_iso_lines(prevalence, costs, prior)

    def roc_over_bayesian_iso_line(x):
        # this function is NOT vectorized
        function_of_x = approximate_y(x) - b_iso_line_y(x)
        return function_of_x
    #enddef

    def roc_left_of_bayesian_iso_line(y):
        # this function is NOT vectorized
        # use (1 - ...) in both terms to flip FPR into TNR
        function_of_y = (1 - approximate_x(y)) - (1 - b_iso_line_x(y))
        return function_of_y
    # enddef

    (A_pi_x, A_pi_y), rootVal = findRoot(roc_over_bayesian_iso_line, approximate_y)
    if rootVal is None:
        (A_pi_x, A_pi_y), rootVal = findRoot(roc_left_of_bayesian_iso_line, approximate_x, isVertical=False)
    #endif
    return (A_pi_x, A_pi_y), rootVal
#enddef

def findRoot(roc_minus_b, approx, isVertical=True):
    from scipy.optimize import brentq
    # A_pi_x     = findRoot(lambda x: roc_over_bayesian_iso_line(x), 0.2, method='broyden1').x
    if isVertical:
        type='vertical'
    else:
        type='horizontal'
    print(f'\nSeeking {type} root on interval [0, 1] with range [{roc_minus_b(0):0.2f}, {roc_minus_b(1):0.2f}]')
    try:
        A_pi_val, r = brentq(lambda z: roc_minus_b(z), 0, 1, full_output=True, disp=False)
        rootFound = r.converged
        if not rootFound:
            print('    Root finding for intersection did not converge')
    except ValueError as e:
        rootFound = False
        print(f'    Root finding for intersection encountered a ValueError:\n    {e}')
    # endtry
    if rootFound:
        if isVertical:
            A_pi_x = A_pi_val
            A_pi_y = float(approx(A_pi_x))
            rootVal = roc_minus_b(A_pi_x)
            print(f'returning vertical root value {rootVal:0.2g} at ({A_pi_x:0.2f},{A_pi_y:0.2f})')
            return (A_pi_x, A_pi_y), rootVal
        else:
            A_pi_y = A_pi_val
            A_pi_x = float(approx(A_pi_y))
            rootVal = roc_minus_b(A_pi_y)
            print(f'returning horizontal root value {rootVal:0.2g} at ({A_pi_x:0.2f},{A_pi_y:0.2f})')
            return (A_pi_x, A_pi_y), rootVal
        # endif
    else:
        print(f'returning no root value')
        return (None, None), None
    # endif
# enddef

def BayesianAUC(fpr, tpr, group, prevalence, costs, prior):
    # this function computes measures over the group/region specified in fpr and tpr
    # that is, the measures are group measures, unless the input fpr and tpr represent the whole

    approximate_y = interp1d(fpr, tpr)  # x,y; to estimate y=f(x)
    approximate_x = interp1d(tpr, fpr)  # y,x; to estimate x=f_inverse(y)

    b_iso_line_y,  b_iso_line_x  = bayesian_iso_lines(prevalence, costs, prior)

    def roc_over_dpos(x):
        # this function is NOT vectorized
        res = approximate_y(x) - x
        return max(res, 0)  # return semi-positive result
    #enddef

    def roc_over_bayesian_iso_line_positive(x):
        # this function is NOT vectorized
        res = approximate_y(x) - b_iso_line_y(x)
        return max(res, 0)  # return semi-positive result
    #enddef

    def roc_over_dneg(x):
        # this function is NOT vectorized
        res = approximate_y(x) - x
        return min(res, 0)  # return semi-negative result
    #enddef

    def roc_over_bayesian_iso_line_negative(x):
        # this function is NOT vectorized
        res = approximate_y(x) - b_iso_line_y(x)
        return min(res, 0)  # return semi-negative result
    #enddef

    def roc_left_of_dpos(y):
        # this function is NOT vectorized
        # use (1 - ...) in both terms to flip FPR into TNR
        res = (1-approximate_x(y)) - (1-y)
        return max(res, 0)  # return semi-positive result
    #enddef

    def roc_left_of_bayesian_iso_line_positive(y):
        # this function is NOT vectorized
        # use (1 - ...) in both terms to flip FPR into TNR
        res = (1-approximate_x(y)) - (1-b_iso_line_x(y))
        return max(res, 0)  # return semi-positive result
    #enddef

    def roc_left_of_dneg(y):
        # this function is NOT vectorized
        # use (1 - ...) in both terms to flip FPR into TNR
        res = (1-approximate_x(y)) - (1-y)
        return min(res, 0)  # return semi-negative result
    #enddef

    def roc_left_of_bayesian_iso_line_negative(y):
        # this function is NOT vectorized
        # use (1 - ...) in both terms to flip FPR into TNR
        res = (1-approximate_x(y)) - (1-b_iso_line_x(y))
        return min(res, 0)  # return semi-negative result
    #enddef

    warnings.filterwarnings('ignore')  # avoid an annoying integration warning.

    # print(f"group: {group['x1']:0.3f}, {group['x2']:0.3f}")
    pAUC_pi_pos  = integrate.quad(lambda x: roc_over_bayesian_iso_line_positive(x), group['x1'], group['x2'])[0]
    pAUC_pi_neg  = integrate.quad(lambda x: roc_over_bayesian_iso_line_negative(x), group['x1'], group['x2'])[0]
    pi_y         = integrate.quad(lambda x: b_iso_line_y(x), group['x1'], group['x2'])[0]
    pAUC_pi      = pAUC_pi_pos + pAUC_pi_neg

    pAUC_d_pos   = integrate.quad(lambda x: roc_over_dpos(x), group['x1'], group['x2'])[0]
    pAUC_d_neg   = integrate.quad(lambda x: roc_over_dneg(x), group['x1'], group['x2'])[0]
    d_y          = integrate.quad(lambda x: x, group['x1'], group['x2'])[0]
    pAUC_d       = pAUC_d_pos + pAUC_d_neg

    # IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
    #   If increasing the limit yields no improvement it is advised to analyze
    #   the integrand in order to determine the difficulties.  If the position of a
    #   local difficulty can be determined (singularity, discontinuity) one will
    #   probably gain from splitting up the interval and calling the integrator
    #   on the subranges.  Perhaps a special-purpose integrator should be used.
    #   pAUCx_pi_pos = integrate.quad(lambda y: roc_left_of_bayesian_iso_line_positive(y), group['y1'], group['y2'])[0]
    pAUCx_pi_pos = integrate.quad(lambda y: roc_left_of_bayesian_iso_line_positive(y), group['y1'], group['y2'])[0]
    pAUCx_pi_neg = integrate.quad(lambda y: roc_left_of_bayesian_iso_line_negative(y), group['y1'], group['y2'])[0]
    pi_x         = integrate.quad(lambda y: 1-b_iso_line_x(y), group['y1'], group['y2'])[0]
    pAUCx_pi     = pAUCx_pi_pos + pAUCx_pi_neg

    pAUCx_d_pos = integrate.quad(lambda y: roc_left_of_dpos(y), group['y1'], group['y2'])[0]
    pAUCx_d_neg = integrate.quad(lambda y: roc_left_of_dneg(y), group['y1'], group['y2'])[0]
    d_x         = integrate.quad(lambda y: 1-y, group['y1'], group['y2'])[0]
    pAUCx_d     = pAUCx_d_pos + pAUCx_d_neg
    warnings.resetwarnings()

    AUCi_pi      = (0.5 * pAUC_pi) + (0.5 * pAUCx_pi)
    AUCi_d       = (0.5 * pAUC_d)  + (0.5 * pAUCx_d)

    delx         = group['x2'] - group['x1']
    dely         = group['y2'] - group['y1']
    # pdelx        = delx - pi_y   # delx*1 is a vertical   area, minus pi_y, a baseline vertical   area
    # pdely        = dely - pi_x   # dely*1 is a horizontal area, minus pi_x, a baseline horizontal area
    # ddelx        = delx - d_y    # delx*1 is a vertical   area, minus d_y,  a baseline vertical   area
    # ddely        = dely - d_x    # dely*1 is a horizontal area, minus d_x,  a baseline horizontal area
    try:
        if   delx!=0 and dely!=0:      # normal case, general equation
            AUCni_pi  = (delx / (delx + dely)) * pAUC_pi/delx + (dely / (delx + dely)) * pAUCx_pi/dely
            pAUCn_pi  = pAUC_pi  / delx
            pAUCxn_pi = pAUCx_pi / dely
            AUCni_d   = (delx / (delx + dely)) * pAUC_d /delx + (dely / (delx + dely)) * pAUCx_d/dely
            pAUCn_d   = pAUC_d   / delx
            pAUCxn_d  = pAUCx_d  / dely
        elif delx==0 and dely!=0:
            AUCni_pi  = pAUCx_pi / dely  # equation reduced for special case
            pAUCn_pi  = 0
            pAUCxn_pi = pAUCx_pi / dely
            AUCni_d   = pAUCx_d  / dely
            pAUCn_d   = 0
            pAUCxn_d  = pAUCx_d  / dely
        elif delx!=0 and dely==0:
            AUCni_pi  = pAUC_pi  / delx   # equation reduced for special case
            pAUCn_pi  = pAUC_pi  / delx
            pAUCxn_pi = 0
            AUCni_d   = pAUC_d   / delx
            pAUCn_d   = pAUC_d   / delx
            pAUCxn_d  = 0
        elif delx==0 and dely==0:
            AUCni_pi  = 0
            pAUCn_pi  = 0
            pAUCxn_pi = 0
        #endif
    except RuntimeError as e:
        print('RuntimeError computing AUCni_pi')
    #endtry

    measures_dict = dict(#AUC_pi=AUC_pi,        # an overall measure
                         #AUC_pi_pos=AUC_pi_pos,    AUC_pi_neg=AUC_pi_neg,
                         AUCi_pi    =AUCi_pi,       # group measure, not normalized, should sum to the overall measure
                         AUCni_pi   =AUCni_pi,      # group measure, normalized
                         pAUC_pi    =pAUC_pi,       pAUCx_pi    =pAUCx_pi,
                         pAUCn_pi   =pAUCn_pi,      pAUCxn_pi   =pAUCxn_pi,
                         pAUC_pi_pos=pAUC_pi_pos,   pAUCx_pi_pos=pAUCx_pi_pos,
                         pAUC_pi_neg=pAUC_pi_neg,   pAUCx_pi_neg=pAUCx_pi_neg,
                         #AUC_d=AUC_d,         # an overall measure
                         #AUC_d_pos=AUC_d_pos,       AUC_d_neg=AUC_d_neg,
                         AUCi_d     =AUCi_d,         # group measure, not normalized, should sum to the overall measure
                         AUCni_d    =AUCni_d,        # group measure, normalized
                         pAUC_d     =pAUC_d,        pAUCx_d     =pAUCx_d,  # not normalized
                         pAUCn_d    =pAUCn_d,       pAUCxn_d    =pAUCxn_d,  # not normalized
                         pAUC_d_pos =pAUC_d_pos,    pAUCx_d_pos =pAUCx_d_pos,
                         pAUC_d_neg =pAUC_d_neg,    pAUCx_d_neg =pAUCx_d_neg)
    return measures_dict




#enddef

def showBayesianAUCmeasures(i, m, groupsArePerfectCoveringSet):
    if groupsArePerfectCoveringSet:
        print(f"cpAUC_{i}_pi{' ':6s} =  {m['cpAUCi_pi']:0.4f}  (not normalized, sums to AUC_pi for this covering set of groups)")
    else:
        print(f"cpAUC_{i}_pi{' ':6s} =  {m['cpAUCi_pi']:0.4f}  (not normalized)")
    #endif
    print(f"AUC_{i}_pi{' ':8s} =  {m['AUCi_pi']:0.4f}  (normalized version of cpAUC_{i}_pi)")
    print(' ')
    print(f"pAUC_{i}_pi+{' ':6s} =  {m['pAUC_pi_pos']:0.4f}  (not normalized)")
    print(f"pAUC_{i}_pi-{' ':6s} = -{abs(m['pAUC_pi_neg']):0.4f}  (not normalized)")
    print(f"pAUC_{i}_pi{' ':7s} =  {m['pAUC_pi']:0.4f}  (not normalized)")
    print(' ')
    print(f"pAUCx_{i}_pi+{' ':5s} =  {m['pAUCx_pi_pos']:0.4f}  (not normalized)")
    print(f"pAUCx_{i}_pi-{' ':5s} = -{abs(m['pAUCx_pi_neg']):0.4f}  (not normalized)")
    print(f"pAUCx_{i}_pi{' ':6s} =  {m['pAUCx_pi']:0.4f}  (not normalized)")
    print(' ')
    return
#enddef