from openai import OpenAI
import json
import random
import time

# Set up variables
credFile = open('../cred.json')
credDict = json.load(credFile)
client = OpenAI(api_key=credDict['api-key'])
num = 120
random.seed(116)

# Get VLAT questions and answers JSON
f = open('../reVLAT_QAs.json')
qaDict = json.load(f)

qaShuffleList = []
for chartKey in qaDict.keys():
    chartDict = qaDict[chartKey]
    plotURL = chartDict['url']
    chartQADict = chartDict['QAs']

    for qaKey in chartQADict.keys():
        questionDict = chartQADict[qaKey]
        questionStr = questionDict['Q']
        answerStr = questionDict['A']

        for i in range(num):
            qaShuffleList.append({'chartKey':chartKey,
                                  'qaKey':qaKey,
                                  'qaInput':questionStr,
                                  'plotURL':plotURL,
                                  'answer':answerStr})


random.shuffle(qaShuffleList)

#for inputDict in qaShuffleList:
for i in range(len(qaShuffleList)):
    inputDict = qaShuffleList[i]

    chartKey = inputDict['chartKey'] 
    qaKey = inputDict['qaKey']
    qaInput = inputDict['qaInput']
    plotURL = inputDict['plotURL']


    if ((chartKey == 'Stacked Bar Chart' and (qaKey == '4' or qaKey == '5')) or
        (chartKey == 'Pie Chart' and (qaKey == '1' or qaKey == '3')) or 
        (chartKey == 'Bubble Chart' and qaKey == '7') or 
        (chartKey == 'Area Chart' and qaKey == '1') or 
        (chartKey == 'Stacked Area Chart' and qaKey == '2') or
        (chartKey == 'Choropleth Map' and qaKey == '3') or
        (chartKey == 'Treemap' and (qaKey == '2' or qaKey == '3'))):
        print('Results/ChatGPT/'+str(i)+'_'+chartKey+'_'+qaKey+'.json')
        start = time.time()
        response = client.chat.completions.create(model='gpt-4-turbo-preview',
                                                messages=[{'role': 'system',
                                                            'content': [{'type': 'text',
                                                                        'text': 'You are a helpful assistant for answering questions. Please answer with the best response in one word.'}]},
                                                            {'role': 'user',
                                                            'content': [{'type': 'text',
                                                                        'text': qaInput}]}],
                                                max_tokens=300)
        end = time.time()
        responseStr = response.choices[0].message.content

        inputDict['response'] = responseStr
        inputDict['time'] = end-start
        with open('Results/ChatGPT/'+str(i)+'_'+chartKey+'_'+qaKey+'.json', 'w') as outfile:
            json.dump(inputDict, outfile)