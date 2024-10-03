from openai import OpenAI
import itertools
import json
import random
import time

# Set up variables
credFile = open('../cred.json')
credDict = json.load(credFile)
client = OpenAI(api_key=credDict['api-key'])
letterList = ['(a)','(b)','(c)','(d)']
num = 120
random.seed(47)

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
        choiceList = questionDict['Choices']
        answerStr = questionDict['A']
        
        permList = list(itertools.permutations(choiceList))
        
        permNum = 0
        for choiceArr in permList:
            qaDict[chartKey]['QAs'][qaKey]['perm'+str(permNum)] = {}
            for i in range(num//len(permList)):
                queryList = [letterList[j]+' '+choice for j,choice in enumerate(choiceArr)]
                queryStr = '\n'.join(queryList)
                print(questionStr+'\n\n'+queryStr)

                qaShuffleList.append({'chartKey':chartKey,
                                      'qaKey':qaKey,
                                      'qaInput':questionStr+'\n\n'+queryStr,
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

    if chartKey == 'Scatterplot' and qaKey == '4':
        print('Results/ChatGPT/'+str(i)+'_'+chartKey+'_'+qaKey+'.json')
        failBool = True
        while failBool:
            try:
                start = time.time()
                response = client.chat.completions.create(model='gpt-4-vision-preview',
                                                        messages=[{'role': 'system',
                                                                    'content': [{'type': 'text',
                                                                                'text': 'You are a helpful assistant for analyzing data visualizations. Please answer with the letter corresponding to the best option, or make a random guess if unsure. For example, if option (a) is correct, only reply with (a).'}]},
                                                                    {'role': 'user',
                                                                    'content': [{'type': 'text',
                                                                                'text': qaInput},
                                                                                {'type': 'image_url',
                                                                                'image_url': plotURL,}]}],
                                                        max_tokens=300)
                end = time.time()
                failBool = False
            except:
                print('Retry in 5 seconds',)
                time.sleep(5)
        
        responseStr = response.choices[0].message.content
        letterAnswer = [l for l in letterList if l in responseStr]

        inputDict['response'] = responseStr
        inputDict['letterResp'] = letterAnswer
        inputDict['time'] = end-start
        with open('Results/ChatGPT/'+str(i)+'_'+chartKey+'_'+qaKey+'.json', 'w') as outfile:
            json.dump(inputDict, outfile)