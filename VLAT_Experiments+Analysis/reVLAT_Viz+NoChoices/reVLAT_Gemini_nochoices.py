import google.generativeai as genai
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

model = genai.GenerativeModel(model_name='gemini-pro-vision',
                              generation_config={'max_output_tokens':300},
                              safety_settings=safety_settings)
num = 120
random.seed(500)

# Get VLAT questions and answers JSON
f = open('../reVLAT_QAs.json')
qaDict = json.load(f)

qaShuffleList = []
for chartKey in qaDict.keys():
    chartDict = qaDict[chartKey]
    plotURL = chartDict['url']
    chartQADict = chartDict['QAs']

    plotPath = chartDict['path']
    img = PIL.Image.open('../'+plotPath)

    for qaKey in chartQADict.keys():
        questionDict = chartQADict[qaKey]
        questionStr = questionDict['Q']
        answerStr = questionDict['A']

        for i in range(num):
            qaShuffleList.append({'chartKey':chartKey,
                                  'qaKey':qaKey,
                                  'qaInput':questionStr,
                                  'plotURL':plotURL,
                                  'img':img,
                                  'answer':answerStr})


random.shuffle(qaShuffleList)

#for inputDict in qaShuffleList:
for i in range(len(qaShuffleList)):
    inputDict = qaShuffleList[i]

    chartKey = inputDict['chartKey'] 
    qaKey = inputDict['qaKey']
    qaInput = inputDict['qaInput']
    plotURL = inputDict['plotURL']
    img = inputDict['img']

    print('Results/Gemini/'+str(i)+'_'+chartKey+'_'+qaKey+'.json')
    prompt = 'You are a helpful assistant for analyzing data visualizations. Please answer with the best response in one word.'
    failBool = True
    while failBool:
        try:
            start = time.time()
            response = model.generate_content([prompt,
                                               qaInput,
                                               img])
            end = time.time()
            response.resolve()
            responseStr = response.text
            failBool = False
        except:
            print('Retry in 5 seconds',)
            time.sleep(5)

    inputDict.pop('img', None)
    inputDict['response'] = responseStr
    inputDict['time'] = end-start
    with open('Results/Gemini/'+str(i)+'_'+chartKey+'_'+qaKey+'.json', 'w') as outfile:
        json.dump(inputDict, outfile)