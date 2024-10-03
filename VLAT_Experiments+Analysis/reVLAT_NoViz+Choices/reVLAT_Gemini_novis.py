import google.generativeai as genai
import itertools
import json
import random
import time
import PIL.Image

# Set up variables
credFile = open('../cred.json')
credDict = json.load(credFile)
genai.configure(api_key=credDict['google-api-key'])

safety_settings = [
    {
        'category': 'HARM_CATEGORY_DANGEROUS',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE',
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE',
    },
]

model = genai.GenerativeModel(model_name='gemini-pro',
                              generation_config={'max_output_tokens':300},
                              safety_settings=safety_settings)
letterList = ['(a)','(b)','(c)','(d)']
num = 120
random.seed(80155)

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

    print('Results/Gemini/'+str(i)+'_'+chartKey+'_'+qaKey+'.json')
    prompt = 'You are a helpful assistant for answering questions. Please answer with the letter corresponding to the best option, or make a random guess if unsure. For instance, if option (a) is correct, please reply with (a).'
    failBool = True
    while failBool:
        try:
            start = time.time()
            response = model.generate_content([prompt,
                                               qaInput])
            end = time.time()
            response.resolve()
            responseStr = response.text
            failBool = False
        except:
            print('Retry in 5 seconds',)
            time.sleep(5)
            
    letterAnswer = [l for l in letterList if l in responseStr]

    inputDict['response'] = responseStr
    inputDict['letterResp'] = letterAnswer
    inputDict['time'] = end-start
    with open('Results/Gemini/'+str(i)+'_'+chartKey+'_'+qaKey+'.json', 'w') as outfile:
        json.dump(inputDict, outfile)