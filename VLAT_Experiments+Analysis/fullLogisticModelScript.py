import numpy as np
import pandas as pd
import math
from multiprocessing import Pool

import plotly.express as px

from analysisFuncs import logisticModRun,logisticTests

dirLoc = 'Analysis/fullLogisticModel'

corrLogisticDF = pd.read_csv(dirLoc+'/logisticData.csv')

randSeedList = [0,47,69,420,80085,23,70,92,443,80108]
cList = np.logspace(-3,3,7)
penaltyList = [{'penalty':'l1',
                'solvers':['liblinear']},
               {'penalty':'l2',
                'solvers':['lbfgs','liblinear','newton-cg','newton-cholesky','sag']},
               {'penalty':None,
                'solvers':['lbfgs','newton-cg','newton-cholesky','sag','saga']},
               {'penalty':'elasticnet',
                'solvers':['saga']}]
l1RatioList = np.linspace(0,1,10)
maxIter = 10000

inputList = logisticTests(corrLogisticDF,
                          dirLoc+'/logisticData.csv',
                          randSeedList,
                          cList,
                          penaltyList,
                          l1RatioList,
                          maxIter)

if __name__ == '__main__':
    pool = Pool()
    results = pool.map(logisticModRun,
                       inputList)
    
    row = 0
    logisticResultDF = pd.DataFrame()
    for result in results:
        paramDict = result[0]
        metricsDict = result[1]
        kfoldNum = result[2]
        
        logisticResultDF.at[row,'penalty'] = paramDict['penalty']
        logisticResultDF.at[row,'solver'] = paramDict['solver']
        logisticResultDF.at[row,'randSeed'] = paramDict['random_state']
        logisticResultDF.at[row,'kfoldNum'] = kfoldNum
        logisticResultDF.at[row,'auc_roc'] = metricsDict['auc_roc']
        logisticResultDF.at[row,'auc_prc'] = metricsDict['auc_prc']
        logisticResultDF.at[row,'f1_score'] = metricsDict['f1_score']
        logisticResultDF.at[row,'aps'] = metricsDict['aps']
        logisticResultDF.at[row,'score'] = metricsDict['score']
        
        if 'C' in paramDict.keys():
            logisticResultDF.at[row,'cVal'] = paramDict['C']
        
        if 'l1_ratio' in paramDict.keys():
            logisticResultDF.at[row,'l1Ratio'] = paramDict['l1_ratio']
        
        row += 1
    
    logisticResultDF.to_csv(dirLoc+'/logisticScores.csv',
                            index=False)
    
    logisticResultDF['penalty'] = logisticResultDF['penalty'].apply(lambda x:x if type(x) is str else '<None>')

    metricList = [{'col':'auc_roc',
                   'title':'AUC ROC Results'},
                  {'col':'auc_prc',
                   'title':'AUC PRC Results'},
                  {'col':'f1_score',
                   'title':'F1 Scores'},
                  {'col':'aps',
                   'title':'Average Precision Scores'},
                  {'col':'score',
                   'title':'Accuracy Scores'}]

    # Plot boxplots
    for metricDict in metricList:
        fig = px.box(logisticResultDF,
                    x='penalty',
                    y=metricDict['col'],
                    title=metricDict['title']+' across All Penalty Types')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_penaltyTypes.html')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_penaltyTypes.png',
                        width=2000,
                        height=1000)

        fig = px.box(logisticResultDF,
                    x='penalty',
                    y=metricDict['col'],
                    color='solver',
                    title=metricDict['title']+' across All Penalty Types with Solvers')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_penaltyTypes+solvers.html')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_penaltyTypes+solvers.png',
                        width=2000,
                        height=1000)
        
        l1Fil = logisticResultDF['penalty']=='l1'
        filDF = logisticResultDF.loc[l1Fil,:]
        filDF['logCVal'] = filDF['cVal'].apply(lambda x:math.log(x,10))
        fig = px.box(filDF,
                    x='logCVal',
                    y=metricDict['col'],
                    title=r'$l_1\text{ Penalty '+metricDict['title']+r' across }\log(C)$')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_l1.html',
                    include_mathjax='cdn')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_l1.png',
                        width=2000,
                        height=1000)
        
        l2Fil = logisticResultDF['penalty']=='l2'
        filDF = logisticResultDF.loc[l2Fil,:]
        filDF['logCVal'] = filDF['cVal'].apply(lambda x:math.log(x,10))
        fig = px.box(filDF,
                    x='logCVal',
                    y=metricDict['col'],
                    color='solver',
                    title=r'$l_2\text{ Penalty '+metricDict['title']+r' across }\log(C)\text{ and Sovers}$')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_l2.html',
                    include_mathjax='cdn')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_l2.png',
                        width=2000,
                        height=1000)
        
        enFil = logisticResultDF['penalty']=='elasticnet'
        filDF = logisticResultDF.loc[enFil,:]
        filDF['logCVal'] = filDF['cVal'].apply(lambda x:math.log(x,10))
        fig = px.box(filDF,
                    x='logCVal',
                    y=metricDict['col'],
                    color='l1Ratio',
                    title=r'$\text{Elastic Net Penalty '+metricDict['title']+r' across }\log(C)\text{ and }l_1 \text{ Ratio}$')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_elasticnet.html')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_elasticnet.png',
                        width=2000,
                        height=1000)
        
        noneFil = logisticResultDF['penalty']=='<None>'
        filDF = logisticResultDF.loc[noneFil,:]
        filDF['logCVal'] = filDF['cVal'].apply(lambda x:math.log(x,10))
        fig = px.box(filDF,
                    x='solver',
                    y=metricDict['col'],
                    title=metricDict['title']+' of No Penalty with Solvers')
        fig.write_html(dirLoc+'/param_tuning/'+metricDict['col']+'_noPenalty.html')
        fig.write_image(dirLoc+'/param_tuning/'+metricDict['col']+'_noPenalty.png',
                        width=2000,
                        height=1000)