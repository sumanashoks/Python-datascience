#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import random
import string # to process standard python strings
import pandas as pd


# In[2]:


data = pd.read_csv('/Users/apple/Desktop/python/NLP - Classification/CHAT BOT/WikiQA-train.txt', sep="\t", header=None)
data.columns=["qstn","ans","num"]
data["qstn"]=data["qstn"].str.lower()# converts to lowercase
data.head()


# In[3]:


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence): 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response=''
    qstns = data['qstn'].tolist()
    qstns.append(user_response)
    TfidfVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVec.fit_transform(qstns)
    vals = cosine_similarity(tfidf[-1], tfidf)
    print(vals)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    #print(req_tfidf)
    #print(idx)
    if(req_tfidf<0.6):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+data['ans'][idx]
        return robo_response


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    robo_response = ''
    qstns = data['qstn'].tolist()
    TfidfVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVec.fit_transform(qstns + [user_response])
    vals = cosine_similarity(tfidf[-1], tfidf[:-1])  # Exclude the user's response
    idx = vals.argsort()[0][-1]  # Choose the most similar question's index
    req_tfidf = vals[0][idx]
    
    if req_tfidf < 0.6:
        robo_response = "I am sorry! I don't understand you."
    else:
        robo_response = data['ans'][idx]
    
    return robo_response


# In[ ]:


flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    # In case of voice not recognized clearly
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")             # terminating Bot when user says bye
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))  # 1. Basic Greetings Reply
            else:
                print("ROBO: ",end="")
                print(response(user_response))
               # sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


# # voice input and text output

# In[ ]:


import speech_recognition as sr                   # import the library

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    r = sr.Recognizer()                                  # initialize recognizer
    with sr.Microphone() as Microphone:                  # mention source it will be either Microphone or audio files.
        print("Speak Anything :")
        audio = r.listen(Microphone)                     # listen to the source
    try:
        text = r.recognize_google(audio)                 # use recognizer to convert our audio into text part.
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize your voice")    # In case of voice not recognized clearly
    user_response = text
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("ROBO: You are welcome..")             # terminating Bot when user says bye
        else:
            if(greeting(user_response)!=None):
                print("ROBO: "+greeting(user_response))  # 1. Basic Greetings Reply
            else:
                print("ROBO: ",end="")
                print(response(user_response))
               # sent_tokens.remove(user_response)
    else:
        flag=False
        print("ROBO: Bye! take care..")


# In[ ]:


import speech_recognition as sr

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything:")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, could not recognize your voice.")
        return ""
    except sr.RequestError:
        print("Request Failed.")
        return ""

print("ROBO: Hello! I'm Robo. I can answer your queries about Chatbots. Type 'Bye' to exit.")
while True:
    user_response = get_audio()
    
    if user_response == 'bye':
        print("ROBO: Goodbye! Take care.")
        break
    
    if user_response in ['thanks', 'thank you']:
        print("ROBO: You're welcome.")
        break
    
    greeting_reply = greeting(user_response)
    if greeting_reply:
        print("ROBO: " + greeting_reply)  # 1. Basic Greetings Reply
    else:
        bot_response = response(user_response)
        print("ROBO: " + bot_response if bot_response else "I'm sorry, I didn't understand that.")


# #  voice input and voice output

# In[ ]:


import speech_recognition as sr                   # import the library
import pyttsx3
engine = pyttsx3.init()

flag=True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while(flag==True):
    r = sr.Recognizer()                                  # initialize recognizer
    with sr.Microphone() as Microphone:                  # mention source it will be either Microphone or audio files.
        print("Speak Anything :")
        audio = r.listen(Microphone)                     # listen to the source
    try:
        text = r.recognize_google(audio)                 # use recognizer to convert our audio into text part.
        print("You said : {}".format(text))
    except:
        print("Sorry could not recognize your voice")    # In case of voice not recognized clearly
    user_response = text
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            engine.say('You are welcome')
            engine.runAndWait()             # terminating Bot when user says bye
        else:
            if(greeting(user_response)!=None):
                engine.say(greeting(user_response))
                engine.runAndWait()  # 1. Basic Greetings Reply
            else:
                engine.say(response(user_response))
                engine.runAndWait()
               # sent_tokens.remove(user_response)
    else:
        flag=False
        engine.say('Bye! take care')
        engine.runAndWait()


# In[ ]:


import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
flag = True

def get_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak Anything:")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text.lower()
    except sr.UnknownValueError:
        print("Sorry, could not recognize your voice.")
        return ""
    except sr.RequestError:
        print("Request Failed.")
        return ""

print("ROBO: Hello! I'm Robo. I can answer your queries about Chatbots. Type 'Bye' to exit.")

while flag:
    user_response = get_audio()
    
    if user_response == 'bye':
        print("ROBO: Goodbye! Take care.")
        engine.say('Goodbye! Take care.')
        engine.runAndWait()
        break
    
    if user_response in ['thanks', 'thank you']:
        print("ROBO: You're welcome.")
        engine.say('You are welcome.')
        engine.runAndWait()
        break
    
    greeting_reply = greeting(user_response)
    if greeting_reply:
        print("ROBO: " + greeting_reply)  # 1. Basic Greetings Reply
        engine.say(greeting_reply)
        engine.runAndWait()
    else:
        bot_response = response(user_response)
        response_text = bot_response if bot_response else "I'm sorry, I didn't understand that."
        print("ROBO: " + response_text)
        engine.say(response_text)
        engine.runAndWait()


# In[ ]:


pip install google-cloud-speech


# In[ ]:


from google.cloud import speech_v1p1beta1 as speech

client = speech.SpeechClient()

config = {
    "language_code": "en-US"
}

while True:
    with sr.Microphone() as source:
        print("Speak Anything:")
        audio = r.listen(source)

    response = client.recognize(config=config, audio={"content": audio.frame_data})
    for result in response.results:
        print("You said:", result.alternatives[0].transcript.lower())
        # Perform actions based on the recognized text here


# In[ ]:




