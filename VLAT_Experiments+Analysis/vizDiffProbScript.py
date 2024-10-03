import pandas as pd
import scipy.stats as stats
from multiprocessing import Pool

from analysisFuncs import appellFunc, dDiffProp, pDiffProp, qDiffProp

def poolVizDiffTest(varDict):
    return vizDiffTest(**varDict)

def vizDiffTest(**kwargs):
    upperLim = kwargs.get('upperLim')
    dist = kwargs.get('dist')
    key = kwargs.get('key')
    vizCol = kwargs.get('vizCol')
    noVizCol = kwargs.get('noVizCol')
    probDF_dir = kwargs.get('probDF_dir')

    probDF = pd.read_csv(probDF_dir)
    diffSeries = probDF[vizCol]-probDF[noVizCol]

    # Viz distribution check
    bounds = [(0, upperLim), (0, upperLim)]
    vizResults = stats.fit(dist,
                           probDF[vizCol],
                           bounds)
    vizStat_ks, vizPVal_ks = stats.kstest(probDF[vizCol],
                                          'beta',
                                          [vizResults.params.a,
                                           vizResults.params.b])
    # No Viz distribution check
    noVizResults = stats.fit(dist,
                             probDF[noVizCol],
                             bounds)
    noVizStat_ks, noVizPVal_ks = stats.kstest(probDF[noVizCol],
                                              'beta',
                                              [noVizResults.params.a,
                                               noVizResults.params.b])
    print(key)
    print(vizPVal_ks,noVizPVal_ks)
    if (vizPVal_ks <= 0.05)|(noVizPVal_ks <= 0.05):
        betaDiffBool = False
        ecdfResult = stats.ecdf(diffSeries)
        zeroCDF_hat = ecdfResult.cdf.evaluate(0)
        LB_hat = max(ecdfResult.cdf.quantiles[ecdfResult.cdf.probabilities <= 0.025])
        UB_hat = min(ecdfResult.cdf.quantiles[ecdfResult.cdf.probabilities >= 0.975])
        fail2RejectBool = (LB_hat <= 0)&(UB_hat >= 0)
        
        oneSide_LB_hat = max(ecdfResult.cdf.quantiles[ecdfResult.cdf.probabilities <= 0.05])
        oneSide_fail2RejectBool = oneSide_LB_hat <= 0
    else:
        betaDiffBool = True
        zeroCDF_hat = pDiffProp(0,
                                vizResults.params.a,
                                vizResults.params.b,
                                noVizResults.params.a,
                                noVizResults.params.b,
                                dps=15)
        LB_hat = qDiffProp(0.025,
                           vizResults.params.a,
                           vizResults.params.b,
                           noVizResults.params.a,
                           noVizResults.params.b,
                           dps=15)
        UB_hat = qDiffProp(0.975,
                           vizResults.params.a,
                           vizResults.params.b,
                           noVizResults.params.a,
                           noVizResults.params.b,
                           dps=15)
        fail2RejectBool = (LB_hat <= 0)&(UB_hat >= 0)
        
        oneSide_LB_hat = qDiffProp(0.05,
                                   vizResults.params.a,
                                   vizResults.params.b,
                                   noVizResults.params.a,
                                   noVizResults.params.b,
                                   dps=15)
        oneSide_fail2RejectBool = oneSide_LB_hat <= 0
    
    
    output = {'key':key,
              'betaDiffBool':betaDiffBool,
              'zeroCDF_hat':zeroCDF_hat,
              'LB_hat':LB_hat,
              'UB_hat':UB_hat,
              'fail2RejectBool':fail2RejectBool,
              'oneSide_LB_hat':oneSide_LB_hat,
              'oneSide_fail2RejectBool':oneSide_fail2RejectBool}
    return output

if __name__ == '__main__':
    dirLoc = 'Analysis/fullLogisticModel'
    probDF = pd.read_csv(dirLoc+'/probValues.csv')

    probColDict = {}
    for col in probDF.columns:
        colList = col.split('&')
        
        if 'reVLAT_VizChoices' in colList:
            colList.remove('reVLAT_VizChoices')
            colList.sort()
            newCol = '&'.join(colList)
            
            if newCol not in probColDict.keys():
                probColDict[newCol] = {'Viz':col}
            else:
                tempDict = probColDict[newCol]
                tempDict['Viz'] = col
                probColDict[newCol] = tempDict
        elif 'reVLAT_NoVizChoices' in colList:
            colList.remove('reVLAT_NoVizChoices')
            colList.sort()
            newCol = '&'.join(colList)
            
            if newCol not in probColDict.keys():
                probColDict[newCol] = {'NoViz':col}
            else:
                tempDict = probColDict[newCol]
                tempDict['NoViz'] = col
                probColDict[newCol] = tempDict

    varList = []
    upperLim = 100000
    dist = stats.beta
    for key in probColDict:
        vizCol = probColDict[key]['Viz']
        noVizCol = probColDict[key]['NoViz']
        varList.append({'upperLim':upperLim,
                        'dist':dist,
                        'key':key,
                        'vizCol':vizCol,
                        'noVizCol':noVizCol,
                        'probDF_dir':dirLoc+'/probValues.csv'})
    
    pool = Pool()
    results = pool.map(poolVizDiffTest,
                       varList)
    
    vizDiffAnalysisDF = pd.DataFrame()
    row = 0
    for result in results:
        vizDiffAnalysisDF.at[row,'colName'] = result['key']
        vizDiffAnalysisDF.at[row,'betaDiff?'] = result['betaDiffBool']
        vizDiffAnalysisDF.at[row,'zeroCDF_hat'] = result['zeroCDF_hat']
        vizDiffAnalysisDF.at[row,'2-side p-value_hat'] = 2*min(result['zeroCDF_hat'],1-result['zeroCDF_hat'])
        vizDiffAnalysisDF.at[row,'2-side LB_hat (<= 0.025)'] = result['LB_hat']
        vizDiffAnalysisDF.at[row,'2-side UB_hat (>= 0.975)'] = result['UB_hat']
        vizDiffAnalysisDF.at[row,'2-side failToReject'] = result['fail2RejectBool']
        
        vizDiffAnalysisDF.at[row,'1-side p-value_hat'] = result['zeroCDF_hat']
        vizDiffAnalysisDF.at[row,'1-side LB_hat (<= 0.05)'] = result['oneSide_LB_hat']
        vizDiffAnalysisDF.at[row,'1-side failToReject'] = result['oneSide_fail2RejectBool']

        row += 1
    
    vizDiffAnalysisDF.to_csv(dirLoc+'/vizDiffAnalysis.csv',
                             index=False)