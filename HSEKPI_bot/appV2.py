import streamlit as st
import pandas as pd
import nltk
import numpy as np
import string
import warnings
import requests
import pickle
import random
import os
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk import edit_distance

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
from responses import *
from data import *

# Lemmitization

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

vectorizer = TfidfVectorizer(tokenizer=Normalize,stop_words = stopwords.words('french'))



def load_doc(jsonFile):
    with open(jsonFile) as file:
        Json_data = json.loads(file.read())
    return Json_data


#data = load_doc('data.json')
#book = load_doc('book.json')
eclf = VotingClassifier(estimators=[ 
    ('svm', SVC(probability=True,C= 2)),
    ('lr', LogisticRegression(C=100.0)),
    ('rf', RandomForestClassifier(n_estimators=50,max_features= 'log2')),
    ], voting='soft')
#eclf= joblib.load('eclf.pkl')
df = pd.DataFrame(data, columns = ["Text","Intent"])
x = df['Text']
y= df['Intent']
X= vectorizer.fit_transform(x)
eclf.fit(X, y)


# To get responnse

def response(user_response):
    text_test = [user_response]
    X_test = vectorizer.transform(text_test)
    prediction = eclf.predict(X_test)
    reply = random.choice(responses[prediction[0]]['response'])
    return reply


# To get indent
def intent(user_response):
    text_intent = [user_response]
    X_test_intent = vectorizer.transform(text_intent)
    predicted_intent = eclf.predict(X_test_intent)
    intent_predicted = responses[predicted_intent[0]]['intent']
    return intent_predicted

import telegram    
from telegram import Update, ForceReply, Bot,ReplyKeyboardMarkup
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

#import matching table
name_list = pd.read_csv("url_links.csv")



def bot_initialize(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        user_intent = intent(user_response)
        
        if (user_intent !=''):
            if (user_response == '/start'):
                resp = """Salut je suis HSEbot une intelligence artificielle qui t'aide à identifier les dangers et les risques ainsi qu'à les prévenirs.Mon créateur est Dahou Renaud L:https://www.linkedin.com/in/dahou-renaud-louis-8958599a/\n\nComment puis-je t'aider ?\n\nTapez Bye pour quitter."""
                return resp
            
            elif (user_intent == 'salutation'):
                resp = str(random.choice(responses[0]['response'])) + ", comment puis-je vous aider?"
                return resp
        
            elif (user_intent == 'conaissance'):
                resp = str(random.choice(responses[1]['response']))+ ", comment puis-je vous aider?"
                return resp
            
            elif (user_intent == 'fin_conversation'):
                resp = random.choice(responses[2]['response'])
                return resp

            elif (user_intent == 'Merci'):
                resp = random.choice(responses[3]['response'])
                return resp

            elif (user_intent == 'but'):
                resp = random.choice(responses[5]['response'])
                return resp

            elif (user_intent == 'conaissance'):
                resp = random.choice(responses[1]['response'])
                return resp
            
            elif (user_intent == "question"):
                user_response=user_response.lower()
                resp =  response(user_response)
                return resp #+ "\n\n🎁CADEAU🎁\nJe t'offre ce document HSE qui te servira pour tes TBM et répondre à certaines questions dont ma réponse te semble incorrecte je suis une intelligence artificielle et je peux faire des erreurs comme l'humain.😊:\n https://drive.google.com/file/d/10nDPjBZZX82XCQUZIlUCujc0PpYDlWhb/view?usp=sharing"
            
            elif (user_intent == "Doc"):
                user_response=user_response.lower()
                resp =  response(user_response)
                update_name = name_list[name_list['CAT']==resp]
                A=list(update_name['URL'])
                listToStr = '\n'.join(map(str, A))
                return listToStr
            
            else:
                resp = "Désolé je ne comprend pas mon vocabulaire est en amélioration.Envoie ta question à mon créateur @Renaud17" #random.choice(responses[4]['response'])
                return resp
                   
        else:
            flag = False
            resp = "Mais vous ne m'avez posé aucune question"+ ", comment puis-je vous aider?" #random.choice(responses[2]['response'])
            return resp
         
def bot_initialize2(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        user_intent = intent(user_response)
        
        if (user_intent !=''):
            if (user_response == '/start'):
                resp = """Salut je suis HSEbot une intelligence artificielle qui t'aide à identifier les dangers et les risques ainsi qu'à les prévenirs.Mon créateur est Dahou Renaud L:https://www.linkedin.com/in/dahou-renaud-louis-8958599a/\n\nComment puis-je t'aider ?\n\nTapez Bye pour quitter."""
                return resp
            
            elif (user_intent == 'salutation'):
                resp = str(random.choice(responses[0]['response'])) + ", comment puis-je vous aider?"
                return resp
        
            elif (user_intent == 'conaissance'):
                resp = str(random.choice(responses[1]['response']))+ ", comment puis-je vous aider?"
                return resp
            
            elif (user_intent == 'fin_conversation'):
                resp = random.choice(responses[2]['response'])
                return resp

            elif (user_intent == 'Merci'):
                resp = random.choice(responses[3]['response'])
                return resp

            elif (user_intent == 'but'):
                resp = random.choice(responses[5]['response'])
                return resp

            elif (user_intent == 'conaissance'):
                resp = random.choice(responses[1]['response'])
                return resp
            
            elif (user_intent == "question"):
                user_response=user_response.lower()
                resp =  response(user_response)
                return resp #+ "\n\n🎁CADEAU🎁\nJe t'offre ce document HSE qui te servira pour tes TBM et répondre à certaines questions dont ma réponse te semble incorrecte je suis une intelligence artificielle et je peux faire des erreurs comme l'humain.😊:\n https://drive.google.com/file/d/10nDPjBZZX82XCQUZIlUCujc0PpYDlWhb/view?usp=sharing"
            
            elif (user_intent == "Doc"):
                user_response=user_response.lower()
                resp =  response(user_response)
                update_name = name_list[name_list['CAT']==resp]
                A=list(update_name['URL'])
                listToStr = '\n'.join(map(str, A))
                return listToStr
            
            else:
                resp = "Désolé je ne comprend pas mon vocabulaire est en amélioration.Envoie ta question à mon créateur @Renaud17" #random.choice(responses[4]['response'])
                return resp
                   
        else:
            flag = False
            resp = "Mais vous ne m'avez posé aucune question"+ ", comment puis-je vous aider?" #random.choice(responses[2]['response'])
            return resp
        
def get_text():
    user_input2 = st.text_input("Toi: ","Ecrivez ici")
    return user_input2

def start(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        f'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )	
            
def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')



def run_bot(update: Update, _: CallbackContext) -> None:
    replic = update.message.text
    answer = bot_initialize(replic)
    update.message.reply_text(answer)
"""
def main() -> None:
    #Start the bot."""
    #updater = Updater("1836903308:AAFE4kcYQ61hmpiGxJMeRP9B6WuG3DQj-Fk")
"""
    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text, run_bot))

    # Start the Bot
    updater.start_polling()
    updater.idle()
    

if __name__ == '__main__':
    main()
"""
