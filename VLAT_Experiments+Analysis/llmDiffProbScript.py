import pandas as pd
import scipy.stats as stats
from multiprocessing import Pool

from analysisFuncs import appellFunc, dDiffProp, pDiffProp, qDiffProp

def poolLLMDiffTest(varDict):
    return llmDiffTest(**varDict)

def llmDiffTest(**kwargs):
    upperLim = kwargs.get('upperLim')
    dist = kwargs.get('dist')
    key = kwargs.get('key')
    chatGPTCol = kwargs.get('chatGPTCol')
    geminiCol = kwargs.get('geminiCol')
    probDF_dir = kwargs.get('probDF_dir')

    probDF = pd.read_csv(probDF_dir)
    diffSeries = probDF[chatGPTCol]-probDF[geminiCol]

    # ChatGPT distribution check
    bounds = [(0, upperLim), (0, upperLim)]
    chatGPTResults = stats.fit(dist,
                               probDF[chatGPTCol],
                               bounds)
    chatGPT_Stat, chatGPT_PVal = stats.kstest(probDF[chatGPTCol],
                                              'beta',
                                              [chatGPTResults.params.a,
                                               chatGPTResults.params.b])
    
    # Gemini distribution check
    geminiResults = stats.fit(dist,
                              probDF[geminiCol],
                              bounds)
    geminiStat, geminiPVal = stats.kstest(probDF[geminiCol],
                                          'beta',
                                          [geminiResults.params.a,
                                           geminiResults.params.b])
    print(key)
    print(chatGPT_PVal,geminiPVal)
    if (chatGPT_PVal <= 0.05)|(geminiPVal <= 0.05):
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
                                chatGPTResults.params.a,
                                chatGPTResults.params.b,
                                geminiResults.params.a,
                                geminiResults.params.b,
                                dps=15)
        LB_hat = qDiffProp(0.025,
                           chatGPTResults.params.a,
                           chatGPTResults.params.b,
                           geminiResults.params.a,
                           geminiResults.params.b,
                           dps=15)
        UB_hat = qDiffProp(0.975,
                           chatGPTResults.params.a,
                           chatGPTResults.params.b,
                           geminiResults.params.a,
                           geminiResults.params.b,
                           dps=15)
        fail2RejectBool = (LB_hat <= 0)&(UB_hat >= 0)
        
        oneSide_LB_hat = qDiffProp(0.05,
                                   chatGPTResults.params.a,
                                   chatGPTResults.params.b,
                                   geminiResults.params.a,
                                   geminiResults.params.b,
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
        
        if 'ChatGPT' in colList:
            colList.remove('ChatGPT')
            colList.sort()
            newCol = '&'.join(colList)
            
            if newCol not in probColDict.keys():
                probColDict[newCol] = {'ChatGPT':col}
            else:
                tempDict = probColDict[newCol]
                tempDict['ChatGPT'] = col
                probColDict[newCol] = tempDict
        elif 'Gemini' in colList:
            colList.remove('Gemini')
            colList.sort()
            newCol = '&'.join(colList)
            
            if newCol not in probColDict.keys():
                probColDict[newCol] = {'Gemini':col}
            else:
                tempDict = probColDict[newCol]
                tempDict['Gemini'] = col
                probColDict[newCol] = tempDict

    varList = []
    upperLim = 100000
    dist = stats.beta
    for key in probColDict:
        chatGPTCol = probColDict[key]['ChatGPT']
        geminiCol = probColDict[key]['Gemini']
        varList.append({'upperLim':upperLim,
                        'dist':dist,
                        'key':key,
                        'chatGPTCol':chatGPTCol,
                        'geminiCol':geminiCol,
                        'probDF_dir':dirLoc+'/probValues.csv'})
    
    pool = Pool()
    results = pool.map(poolLLMDiffTest,
                       varList)
    
    llmDiffAnalysisDF = pd.DataFrame()
    row = 0
    for result in results:
        llmDiffAnalysisDF.at[row,'colName'] = result['key']
        llmDiffAnalysisDF.at[row,'betaDiff?'] = result['betaDiffBool']
        llmDiffAnalysisDF.at[row,'zeroCDF_hat'] = result['zeroCDF_hat']
        llmDiffAnalysisDF.at[row,'2-side p-value_hat'] = 2*min(result['zeroCDF_hat'],1-result['zeroCDF_hat'])
        llmDiffAnalysisDF.at[row,'2-side LB_hat (<= 0.025)'] = result['LB_hat']
        llmDiffAnalysisDF.at[row,'2-side UB_hat (>= 0.975)'] = result['UB_hat']
        llmDiffAnalysisDF.at[row,'2-side failToReject'] = result['fail2RejectBool']
        
        llmDiffAnalysisDF.at[row,'1-side p-value_hat'] = result['zeroCDF_hat']
        llmDiffAnalysisDF.at[row,'1-side LB_hat (<= 0.05)'] = result['oneSide_LB_hat']
        llmDiffAnalysisDF.at[row,'1-side failToReject'] = result['oneSide_fail2RejectBool']

        row += 1
    
    llmDiffAnalysisDF.to_csv(dirLoc+'/llmDiffAnalysis.csv',
                             index=False)