from db import *
from get import *
from subprocess import call
import streamlit as st
import pandas as pd
import datetime
import re
import base64
from datetime import datetime,date
from datetime import datetime, timedelta
from pandas import DataFrame
from io import BytesIO
import xlsxwriter
import plotly.express as px
from PIL import Image
import streamlit.components.v1 as components

#===================================================
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
#eclf= joblib.load('eclf.pkl')
eclf = VotingClassifier(estimators=[ 
    ('svm', SVC(probability=True,C= 2)),
    ('lr', LogisticRegression(C=100.0)),
    ('rf', RandomForestClassifier(n_estimators=50,max_features= 'auto')),
    ], voting='soft')
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
        f'Salut {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )		
            
def help_command(update: Update, _: CallbackContext) -> None:
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')



def run_bot(update: Update, _: CallbackContext) -> None:
    replic = update.message.text
    answer = bot_initialize(replic)
    update.message.reply_text(answer)


#===================================================


@st.cache(allow_output_mutation=True)
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data
@st.cache(allow_output_mutation=True)
def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Votre fichier excel</a>' # decode b'abc' => abc

#pour verifier le type d'entrée
def inputcheck(inputext):
    try:
        inputext = int(inputext)
    except:
        st.error("Veillez à ne saisir qu'un nombre")
        st.stop()
    return inputext






# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
@st.cache(allow_output_mutation=True)
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

@st.cache(allow_output_mutation=True)
def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False



def main():
    #couleur du select box
    def style():
        st.markdown("""<style>
        div[data-baseweb="select"]> div {
        background-color: yellow;
        } 
        div[role="listbox"] ul {
        background-color:white;
        }</style>""", unsafe_allow_html=True)
        
    #couleur button
    primaryColor = st.get_option("theme.primaryColor")
    s = f"""
    <style>
    div.stButton > button:first-child {{text-shadow:0px 1px 0px #2f6627;font-size:15px; background-color: #71f9ed;border: 5px solid {primaryColor}; border-radius:5px 5px 5px 5px; }}
    <style>
    """
    st.markdown(s, unsafe_allow_html=True)
    
    #masquer le menu streamlit
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    menu = ["Accueil", "Connexion","Inscription"] #
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Accueil":
        components.html("""

			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}

			/* Slideshow container */
			.slideshow-container {
			  max-width: 1000px;
			  position: relative;
			  margin: auto;
			}

			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}

			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}

			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}

			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}

			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}

			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}

			.active, .dot:hover {
			  background-color: #717171;
			}

			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}

			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}

			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}

			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>

			<div class="slideshow-container">

			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://cdn.shopify.com/s/files/1/2382/6729/products/SP124958.jpg?v=1536179866" style="width:100%;border-radius:5px;">
			  <div class="text"></div>
			</div>

			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>

			  <img src="https://www.hsetrain.org/images/slide1.jpg" style="width:100%;border-radius:5px;">
			  <div class="text"></div>
			</div>

			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.spc.com.sg/wp-content/uploads/2015/11/banner-community-society-hse.jpg" style="width:100%;border-radius:5px;">
			  <div class="text"></div>
			</div>

			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>

			</div>
			<br>

			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>

			<script>
			var slideIndex = 1;
			showSlides(slideIndex);

			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}

			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}

			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>


			""")
        html_temp = """
		<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:3px;">
		<h1 style="font-family: 'BadaBoom BB', sans-serif;color:white;text-align:center;"><b>HSE KPI RECORDER & HSEbot</b></h1>
		</div>
		"""
	
        #components.html(html_temp)
        st.markdown(html_temp, unsafe_allow_html = True)
        st.markdown("✨ **Elle est une application d'analyse et de suivi des indicateurs de performance HSE dotée d'une intelligence artificielle pour identifier et prevenir les risques et dangers au travail.**")
        st.markdown("✨ **Vous pouvez ajouter; modifier; supprimer et visualiser vos données avec des graphes.**")
        st.markdown("✨ **Vous pouvez aussi téléchager vos données selon des intervalles de date.**")
        st.markdown("✨ **HSEbot vous permet de discuter de manière inter-active avec une intelligence artificielle qui vous donne des conseils de prévention sur les risques au chantier.**")


        image_BOT = """
		<center><img src="https://www.trainingjournal.com/sites/www.trainingjournal.com/files/styles/original_-_local_copy/entityshare/23924%3Fitok%3DKw_wPH9G"  alt="HSEBOT" height="150" width="200"></center>
		"""
        
        col1, col2, col3 = st.beta_columns([1,10,1])
        with col2:
            st.markdown(image_BOT, unsafe_allow_html = True)	
            #st.image("https://www.trainingjournal.com/sites/www.trainingjournal.com/files/styles/original_-_local_copy/entityshare/23924%3Fitok%3DKw_wPH9G",width=400,)
            #Bot HSE
            user_input3 = get_text()
            response3 = bot_initialize2(user_input3)
            st.text_area("HSEBot:", value=response3, height=200, max_chars=None, key=None)
	

    elif choice == "Connexion":
        st.subheader("Section Connexion")
        email = st.sidebar.text_input("Email")
        password = st.sidebar.text_input("Mot de passe",type='password')
        if st.sidebar.checkbox("Connexion"):
            # if password == '12345':
            create_table()
            hashed_pswd = make_hashes(password)



            result = login_user(email,check_hashes(password,hashed_pswd))
            if result:
                st.success("Connecté en tant que {}".format(email))
                #task = st.selectbox("Task",["Add Post","Analytics","Profiles"])
                task = ""
                if task == "":
                    st.subheader("")
                    
                    image_temp ="""
                    <div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
		    <img src="https://1tpecash.fr/wp-content/uploads/elementor/thumbs/Renaud-Louis-osf6t5lcki4q31uzfafpi9yx3zp4rrq7je8tj6p938.png" alt="Avatar" style="vertical-align: middle;width: 50px;height: 50px;border-radius: 50%;" >
		    <br/>
		    <p style="color:white;text-align:justify">Bienvenue ! Je vous souhaite une bonne expérience, ce travail est le fruit de mes expériences en tant que Manager HSE et Data scientist vos avis à propos sont les bienvenues.</p>
		    </div>
                    """
                    title_temp = """
                	<div style="background-color:#464e5f;padding:10px;border-radius:5px;margin:10px;">
                	<h1 style ="color:white;text-align:center;"> GESTION DES INDICATEURS HSE </h1>
                	</div>
                	"""
                    st.markdown(image_temp, unsafe_allow_html = True)
                    st.markdown(title_temp, unsafe_allow_html = True)
                    #st.markdown('### GESTION DES INDICATEURS HSE')
                    style()
                    choix = st.selectbox("", ["AJOUTER", "AFFICHER", "METTRE À JOUR", "SUPPRIMER"])
                    if choix == "AJOUTER":
                        st.subheader("AJOUTER DES DONNÉES")
                        col1, col2= st.beta_columns(2)
                        with col1:
                            st.subheader("CIBLE A ENREGISTRER")
                            
                            style()
                            cible = st.selectbox('', ['Accueil sécurité','Briefing de sécurité( TBM)','Non conformité','Changements enregistrés','Anomalies','Analyse des risques réalisés(JSA)','Incident & Accident',"Audit-Inspection-Exercice d'urgence"])
                            #connexion à l'interface et recupération des données
                            if cible == 'Accueil sécurité':
                                with col1:
                                    Nbre_Arrivant =inputcheck(st.text_input("Nombre Arrivant",value=0))
                                    Nbre_induction = inputcheck(st.text_input("Nombre d'induction",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button1=st.button("AJOUTER LES DÉTAILS")
                                if button1:
                                    add_Accueil(IDD,Chantier,Nbre_Arrivant,Nbre_induction,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))

                            elif cible == 'Briefing de sécurité( TBM)':
                                with col1:
                                    Nbre_chantier =inputcheck(st.text_input("Nombre de chantier",value=0))
                                    Nbre_TBM = inputcheck(st.text_input("Nombre de TBM",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button2=st.button("AJOUTER LES DÉTAILS")
                                if button2:
                                    add_TBM(IDD,Chantier,Nbre_chantier,Nbre_TBM,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == 'Non conformité':
                                with col1:
                                    NCR = inputcheck(st.text_input("Nombre de Non conformité remontée",value=0,key=0))
                                    FNCR = inputcheck(st.text_input("Nombre de fiche de Non conformité remontée",value=0,key=1))
                                    NCC = inputcheck(st.text_input("Nombre de Non conformité cloturée",value=0,key=2))
                                    FNCC= inputcheck(st.text_input("Nombre de fiche de Non conformité cloturée",value=0, key=3))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button3=st.button("AJOUTER LES DÉTAILS")
                                if button3:
                                    add_NC(IDD,Chantier,NCR,FNCR,NCC,FNCC,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == "Changements enregistrés":
                                with col1:
                                    NCH = inputcheck(st.text_input("Nombre de Changement enregistrés",value=0))
                                    FNCH = inputcheck(st.text_input("Nombre de fiche de Changements enregistrés",value=0))
                                    NCHC  = inputcheck(st.text_input("Nombre de Changements cloturés",value=0))
                                    FNCHC= inputcheck(st.text_input("Nombre de fiche de  Changements suivis et cloturés",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button4=st.button("AJOUTER LES DÉTAILS")
                                if button4:
                                    add_Changements(IDD,Chantier,NCH,FNCH,NCHC,FNCHC,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == "Anomalies":
                                with col1:
                                    NA = inputcheck(st.text_input("Nombre d'Anomalies Remontées",value=0))
                                    FNA = inputcheck(st.text_input("Nombre de fiche d'Anomalies Remontées",value=0))
                                    NAC = inputcheck(st.text_input("Nombre d' Anomalies cloturés",value=0))
                                    FNAC = inputcheck(st.text_input("Nombre de fiche de  Anomalies Corrigées",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button5=st.button("AJOUTER LES DÉTAILS")
                                if button5:
                                    add_Anomalies(IDD,Chantier,NA,FNA,NAC,FNAC,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == "Analyse des risques réalisés(JSA)":
                                with col1:
                                    NAct = inputcheck(st.text_input("Nombre d'Activite",value=0))
                                    NJSA = inputcheck(st.text_input("Nombre de fiche JSA",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button6=st.button("AJOUTER LES DÉTAILS")
                                if button6:
                                    add_JSA(IDD,Chantier,NAct,NJSA,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == "Incident & Accident":
                                with col1:
                                    AAA = inputcheck(st.text_input("Accident Avec Arrêt",value=0))
                                    NJP = inputcheck(st.text_input("Nombre de jours perdus",value=0))
                                    ASA = inputcheck(st.text_input("Accident Sans Arrêt",value=0))
                                    AT = inputcheck(st.text_input("Nombre d'accident de trajet",value=0))
                                    NInc = inputcheck(st.text_input("Incident",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button7=st.button("AJOUTER LES DÉTAILS")
                                if button7:
                                    add_Incident_Accident(IDD,Chantier,NInc,AAA,ASA,AT,NJP,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))
                            elif cible == "Audit-Inspection-Exercice d'urgence":
                                with col1:
                                    AC= inputcheck(st.text_input("Nombre d'audit",value=0))
                                    VC= inputcheck(st.text_input("Nombre de Visite Conjointe",value=0))
                                    NEU= inputcheck(st.text_input("Nombre d'exercice d'urgence",value=0))
                                    SMPAR= inputcheck(st.text_input("Sensibilisation au modes de prévention des activités à risques",value=0))
                                    PR= inputcheck(st.text_input("Procedures réalisées",value=0))
                                    IE= inputcheck(st.text_input("Inspections Environnementales",value=0))
                                    IDD=email
                                with col2:
                                    st.subheader("DATE ET NOM DU CHANTIER")
                                    Date = st.date_input("Date")
                                    Chantier = st.text_input("Chantier")
                                    button8=st.button("AJOUTER LES DÉTAILS")
                                if button8:
                                    add_Audit(IDD,Chantier,AC,VC,NEU,SMPAR,PR,IE,Date)
                                    st.success("AJOUTÉ AVEC SUCCÈS: {}".format(Chantier))

                    #visualisation des données
                    elif choix == "AFFICHER":
                        st.subheader("AFFICHEZ VOS DONNÉES")
                        st.warning("Si vous faites des enregistrements à une date antérieure à celle de votre inscription veuillez spécifier l'intervalle de date, car l'affichage des données est par défaut à partir de votre jour d'inscription.")
                        ACCUEIL_exp= st.beta_expander("ACCUEIL SECURITÉ")
                        with ACCUEIL_exp:
                            df_Accueil = pd.DataFrame(view_Accueil(), columns=["id","IDD","Chantier","Nbre_Arrivant","Nbre_induction","Date"])

                            #pour voir uniquement les donnée de l'user connecté
                            IDD2 = email.strip('][').split(', ')

                            #ACCUEIL
                            @st.cache
                            def Accueil_2(df_Accueil: pd.DataFrame) -> pd.DataFrame:
                                df_Accueil2 = df_Accueil[(df_Accueil["IDD"].isin(IDD2))]
                                return df_Accueil2.loc[1:, ["id","Chantier","Nbre_Arrivant","Nbre_induction","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_Accueil1 = Accueil_2(df_Accueil)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_Accueil1['Date'] = pd.to_datetime(df_Accueil1['Date']).apply(lambda x: x.date())
                            df_Accueil1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_Accueil1['Date']))
                                maxy= st.date_input('MaxDate',max(df_Accueil1['Date']))
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas au moins deux dates enregistrées.")
                                st.stop()


                            #filtrage par chantier
                            splitted_df_Accueil1 = df_Accueil1['Chantier'].str.split(',')
                            unique_vals1 = list(dict.fromkeys([y for x in splitted_df_Accueil1  for y in x]).keys())
                            filtrechantier = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals1,key=0)
                            
                            mask = (df_Accueil1['Date'] > miny) & (df_Accueil1['Date'] <= maxy) & (df_Accueil1['Chantier'] == filtrechantier)
                            df_filter1=df_Accueil1.loc[mask]
                            st.dataframe(df_filter1)
                            st.text("*Nbre_Arrivant: Nombre d'arrivant\n*Nbre_induction: Nombre d'induction")

                            if st.button("Télécharger",key=0):
                                st.markdown(get_table_download_link(df_filter1), unsafe_allow_html=True)
                            #figure
                            df_filter1['Nbre_Arrivant'] = pd.to_numeric(df_filter1['Nbre_Arrivant'])
                            df_filter1['Nbre_induction'] = pd.to_numeric(df_filter1['Nbre_induction'])
                            Objectf_fixé= df_filter1['Nbre_Arrivant'].sum()
                            Objectif_atteint = df_filter1['Nbre_induction'].sum()
                            df_filter1_df = pd.DataFrame(columns=["Nombre d'arrivant", "Nombre d'induction"])
                            df_filter1_df.at[0, "Nombre d'arrivant"] = Objectf_fixé
                            df_filter1_df.at[0, "Nombre d'induction"] = Objectif_atteint
                            df_filter1_df_melt = pd.melt(df_filter1_df)
                            df_filter1_df_melt.columns = ['variable', 'valeur']

                            st.dataframe(df_filter1_df_melt)

                            fig = px.bar(df_filter1_df_melt, x = 'variable', y = 'valeur',color="variable")
                            st.plotly_chart(fig, use_container_width=True)


                        BRIEFING_exp= st.beta_expander("BRIEFING DE SÉCURITÉ( TBM)")
                        with BRIEFING_exp:
                            #TMB
                            df_TBM = pd.DataFrame(view_TBM(), columns=["id","IDD","Chantier","Nbre_chantier","Nbre_TBM","Date"])
                            IDD2 = email.strip('][').split(', ')
			    
                            @st.cache
                            def TBM_2(df_TBM: pd.DataFrame) -> pd.DataFrame:
                                df_TBM2 = df_TBM[(df_TBM["IDD"].isin(IDD2))]
                                return df_TBM2.loc[1:, ["id","Chantier","Nbre_chantier","Nbre_TBM","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_TBM1 = TBM_2(df_TBM)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()


                            df_TBM1['Date'] = pd.to_datetime(df_TBM1['Date']).apply(lambda x: x.date())
                            df_TBM1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_TBM1['Date']),key=0)
                                maxy= st.date_input('MaxDate',max(df_TBM1['Date']),key=0)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas au moins deux dates enregistrées.")
                                st.stop()

                            #filtrage par chantier
                            splitted_df_TBM1 = df_TBM1['Chantier'].str.split(',')
                            unique_vals2 = list(dict.fromkeys([y for x in splitted_df_TBM1  for y in x]).keys())
                            filtrechantier2 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals2,key=1)
                            
                            mask = (df_TBM1['Date'] > miny) & (df_TBM1['Date'] <= maxy) & (df_TBM1['Chantier'] == filtrechantier2)
                            df_filter2=df_TBM1.loc[mask]
                            st.dataframe(df_filter2)
                            st.text("*Nbre_chantier: Nombre de chantier\n*Nbre_TBM: Nombre de TBM")

                            if st.button("Télécharger", key=1):
                                st.markdown(get_table_download_link(df_filter2), unsafe_allow_html=True)
                            #figure
                            df_filter2['Nbre_chantier'] = pd.to_numeric(df_filter2['Nbre_chantier'])
                            df_filter2['Nbre_TBM'] = pd.to_numeric(df_filter2['Nbre_TBM'])
                            Objectf_fixé2= df_filter2['Nbre_chantier'].sum()
                            Objectif_atteint2 = df_filter2['Nbre_TBM'].sum()
                            df_filter2_df = pd.DataFrame(columns=["Nombre de chantier", "Nombre de TBM"])
                            df_filter2_df.at[0, "Nombre de chantier"] = Objectf_fixé2
                            df_filter2_df.at[0, "Nombre de TBM"] = Objectif_atteint2
                            df_filter2_df_melt = pd.melt(df_filter2_df)
                            df_filter2_df_melt.columns = ['variable', 'valeur']

                            st.dataframe(df_filter2_df_melt)

                            figTBM = px.bar(df_filter2_df_melt, x = 'variable', y = 'valeur',color="variable")
                            st.plotly_chart(figTBM, use_container_width=True)


                        CONFORMITÉ_exp= st.beta_expander("NON CONFORMITÉ")
                        with CONFORMITÉ_exp:
                            #NON CONFORMITÉ
                            df_NC = pd.DataFrame(view_NC(), columns=["id","IDD","Chantier","NCR","FNCR","NCC","FNCC","Date"])
                            IDD2 = email.strip('][').split(', ')

			    
                            @st.cache
                            def NC_2(df_NC: pd.DataFrame) -> pd.DataFrame:
                                df_NC2 = df_NC[(df_NC["IDD"].isin(IDD2))]
                                return df_NC2.loc[1:, ["id","Chantier","NCR","FNCR","NCC","FNCC","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_NC1 = NC_2(df_NC)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_NC1['Date'] = pd.to_datetime(df_NC1['Date']).apply(lambda x: x.date())
                            df_NC1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_NC1['Date']),key=1)
                                maxy= st.date_input('MaxDate',max(df_NC1['Date']),key=1)

                            except:
                                st.error("Nous ne pouvons afficher car vous n'avez pas aumoins deux dates enrégistrées.")
                                st.stop()



                            #filtrage par chantier
                            splitted_df_NC1 = df_NC1['Chantier'].str.split(',')
                            unique_vals3 = list(dict.fromkeys([y for x in splitted_df_NC1  for y in x]).keys())
                            filtrechantier3 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals3,key=2)
                            
                            mask = (df_NC1['Date'] > miny) & (df_NC1['Date'] <= maxy) & (df_NC1['Chantier'] == filtrechantier3)
                            df_filter3=df_NC1.loc[mask]
                            st.dataframe(df_filter3)
                            st.text("*NCR: Non conformité remontée\n*FNCR: Nombre de fiche de Non conformité remontée\n*NCC: Nombre de Non conformité cloturée\n*FNCC:Nombre de fiche de Non conformité cloturée")


                            if st.button("Télécharger", key=2):
                                st.markdown(get_table_download_link(df_filter3), unsafe_allow_html=True)
                            #figure
                            df_filter3['NCR'] = pd.to_numeric(df_filter3['NCR'])
                            df_filter3['NCC'] = pd.to_numeric(df_filter3['NCC'])
                            df_filter3['FNCR'] = pd.to_numeric(df_filter3['FNCR'])
                            df_filter3['FNCC'] = pd.to_numeric(df_filter3['FNCC'])

                            Objectf_fixe3 = df_filter3['NCR'].sum()
                            Objectif_atteint3 = df_filter3['NCC'].sum()
                            Objectf_fixe4= df_filter3['FNCR'].sum()
                            Objectif_atteint4 = df_filter3['FNCC'].sum()

                            df_filter3_df1 = pd.DataFrame(columns=["NCR", "NCC"])
                            df_filter3_df2 = pd.DataFrame(columns=["FNCR", "FNCC"])

                            df_filter3_df1.at[0, "NCR"] = Objectf_fixe3
                            df_filter3_df1.at[0, "NCC"] = Objectif_atteint3
                            df_filter3_df2.at[0, "FNCR"] = Objectf_fixe4
                            df_filter3_df2.at[0, "FNCC"] = Objectif_atteint4

                            df_filter3_df_melt1 = pd.melt(df_filter3_df1)
                            df_filter3_df_melt2 = pd.melt(df_filter3_df2)

                            df_filter3_df_melt1.columns = ['variable', 'valeur']
                            df_filter3_df_melt2.columns = ['variable', 'valeur']

                            st.dataframe(df_filter3_df_melt1)
                            st.dataframe(df_filter3_df_melt2)

                            figNC1 = px.bar(df_filter3_df_melt1, x = 'variable', y = 'valeur',color="variable")
                            figNC2 = px.bar(df_filter3_df_melt2, x = 'variable', y = 'valeur',color="variable")

                            st.plotly_chart(figNC1, use_container_width=True)
                            st.plotly_chart(figNC2, use_container_width=True)



                        CHANGEMENTS_exp= st.beta_expander("CHANGEMENTS ENREGISTRÉS")
                        with CHANGEMENTS_exp:
                            #CHANGEMENTS
                            df_Changements = pd.DataFrame(view_Changements(), columns=["id","IDD","Chantier","NCH","FNCH","NCHC","FNCHC","Date"])
                            IDD2 = email.strip('][').split(', ')
			
                            @st.cache
                            def Changements_2(df_Changements: pd.DataFrame) -> pd.DataFrame:
                                df_Changements2 = df_Changements[(df_Changements["IDD"].isin(IDD2))]
                                return df_Changements2.loc[1:, ["id","Chantier","NCH","FNCH","NCHC","FNCHC","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_Changements1 = Changements_2(df_Changements)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_Changements1['Date'] = pd.to_datetime(df_Changements1['Date']).apply(lambda x: x.date())
                            df_Changements1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_Changements1['Date']),key=2)
                                maxy= st.date_input('MaxDate',max(df_Changements1['Date']),key=2)
                            except:
                                st.error("Nous ne pouvons afficher car vous n'avez pas aumoins deux dates enrégistrées")
                                st.stop()


                            #filtrage par chantier
                            splitted_df_Changements1 = df_Changements1['Chantier'].str.split(',')
                            unique_vals4 = list(dict.fromkeys([y for x in splitted_df_Changements1  for y in x]).keys())
                            filtrechantier4 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals4,key=3)
                            
                            mask = (df_Changements1['Date'] > miny) & (df_Changements1['Date'] <= maxy) & (df_Changements1['Chantier'] == filtrechantier4)
                            df_filter4=df_Changements1.loc[mask]
                            st.dataframe(df_filter4)
                            st.text("*NCH: Nombre de Changement enregistrés\n*FNCH: Nombre de fiche de Changements enregistrés\n*NCHC: Nombre de Changements cloturés\n*FNCHC:Nombre de fiche de Changements suivis et cloturés")



                            if st.button("Télécharger", key=3):
                                st.markdown(get_table_download_link(df_filter4), unsafe_allow_html=True)
                            #figure
                            df_filter4['NCH'] = pd.to_numeric(df_filter4['NCH'])
                            df_filter4['NCHC'] = pd.to_numeric(df_filter4['NCHC'])
                            df_filter4['FNCH'] = pd.to_numeric(df_filter4['FNCH'])
                            df_filter4['FNCHC'] = pd.to_numeric(df_filter4['FNCHC'])

                            Objectf_fixe4 = df_filter4['NCH'].sum()
                            Objectif_atteint4 = df_filter4['NCHC'].sum()
                            Objectf_fixe5= df_filter4['FNCH'].sum()
                            Objectif_atteint5 = df_filter4['FNCHC'].sum()

                            df_filter4_df1 = pd.DataFrame(columns=["NCH", "NCHC"])
                            df_filter4_df2 = pd.DataFrame(columns=["FNCH", "FNCHC"])

                            df_filter4_df1.at[0, "NCH"] = Objectf_fixe4
                            df_filter4_df1.at[0, "NCHC"] = Objectif_atteint4
                            df_filter4_df2.at[0, "FNCH"] = Objectf_fixe5
                            df_filter4_df2.at[0, "FNCHC"] = Objectif_atteint5

                            df_filter4_df_melt1 = pd.melt(df_filter4_df1)
                            df_filter4_df_melt2 = pd.melt(df_filter4_df2)

                            df_filter4_df_melt1.columns = ['variable', 'valeur']
                            df_filter4_df_melt2.columns = ['variable', 'valeur']

                            st.dataframe(df_filter4_df_melt1)
                            st.dataframe(df_filter4_df_melt2)

                            figCH1 = px.bar(df_filter4_df_melt1, x = 'variable', y = 'valeur',color="variable")
                            figCH2 = px.bar(df_filter4_df_melt2, x = 'variable', y = 'valeur',color="variable")

                            st.plotly_chart(figCH1, use_container_width=True)
                            st.plotly_chart(figCH2, use_container_width=True)

                        ANOMALIES_exp= st.beta_expander("ANOMALIES")
                        with ANOMALIES_exp:
                            #ANOMALIES
                            df_Anomalies = pd.DataFrame(view_Anomalies(), columns=["id","IDD","Chantier","NA","FNA","NAC","FNAC","Date"])
                            IDD2 = email.strip('][').split(', ')
			    
                            @st.cache
                            def Anomalies_2(df_Anomalies: pd.DataFrame) -> pd.DataFrame:
                                df_Anomalies2 = df_Anomalies[(df_Anomalies["IDD"].isin(IDD2))]
                                return df_Anomalies2.loc[1:, ["id","Chantier","NA","FNA","NAC","FNAC","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_Anomalies1 = Anomalies_2(df_Anomalies)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_Anomalies1['Date'] = pd.to_datetime(df_Anomalies1['Date']).apply(lambda x: x.date())
                            df_Anomalies1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_Anomalies1['Date']),key=3)
                                maxy= st.date_input('MaxDate',max(df_Anomalies1['Date']),key=3)
                            except:
                                st.error("Nous ne pouvons afficher car vous n'avez pas aumoins deux dates enrégistrées")
                                st.stop()


                            #filtrage par chantier
                            splitted_df_Anomalies1 = df_Anomalies1['Chantier'].str.split(',')
                            unique_vals5 = list(dict.fromkeys([y for x in splitted_df_Anomalies1  for y in x]).keys())
                            filtrechantier5 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals5,key=4)
                            
                            mask = (df_Anomalies1['Date'] > miny) & (df_Anomalies1['Date'] <= maxy) & (df_Anomalies1['Chantier'] == filtrechantier5)
                            df_filter5=df_Anomalies1.loc[mask]
                            st.dataframe(df_filter5)
                            st.text("*NA: Nombre d'anomalies enregistrés\n*FNA: Nombre de fiche d'anomalies enregistrés\n*NAC: Nombre d'anomalies Corrigées\n*FNAC:Nombre de fiche d'anomalies Corrigées")

                            if st.button("Télécharger", key=4):
                                st.markdown(get_table_download_link(df_filter5), unsafe_allow_html=True)
                            #figure
                            df_filter5['NA'] = pd.to_numeric(df_filter5['NA'])
                            df_filter5['NAC'] = pd.to_numeric(df_filter5['NAC'])
                            df_filter5['FNA'] = pd.to_numeric(df_filter5['FNA'])
                            df_filter5['FNAC'] = pd.to_numeric(df_filter5['FNAC'])

                            Objectf_fixe5 = df_filter5['NA'].sum()
                            Objectif_atteint5 = df_filter5['NAC'].sum()
                            Objectf_fixe6= df_filter5['FNA'].sum()
                            Objectif_atteint6 = df_filter5['FNAC'].sum()

                            df_filter5_df1 = pd.DataFrame(columns=["NA", "NAC"])
                            df_filter5_df2 = pd.DataFrame(columns=["FNA", "FNAC"])

                            df_filter5_df1.at[0, "NA"] = Objectf_fixe5
                            df_filter5_df1.at[0, "NAC"] = Objectif_atteint5
                            df_filter5_df2.at[0, "FNA"] = Objectf_fixe6
                            df_filter5_df2.at[0, "FNAC"] = Objectif_atteint6

                            df_filter5_df_melt1 = pd.melt(df_filter5_df1)
                            df_filter5_df_melt2 = pd.melt(df_filter5_df2)

                            df_filter5_df_melt1.columns = ['variable', 'valeur']
                            df_filter5_df_melt2.columns = ['variable', 'valeur']

                            st.dataframe(df_filter5_df_melt1)
                            st.dataframe(df_filter5_df_melt2)

                            figNA1 = px.bar(df_filter5_df_melt1, x = 'variable', y = 'valeur',color="variable")
                            figNA2 = px.bar(df_filter5_df_melt2, x = 'variable', y = 'valeur',color="variable")

                            st.plotly_chart(figNA1, use_container_width=True)
                            st.plotly_chart(figNA2, use_container_width=True)

                        ANALYSE_exp= st.beta_expander("ANALYSE DES RISQUES RÉALISÉS(JSA)")
                        with ANALYSE_exp:
                            #JSA
                            df_JSA = pd.DataFrame(view_JSA(), columns=["id","IDD","Chantier","NAct","NJSA","Date"])
                            IDD2 = email.strip('][').split(', ')
			    
                            @st.cache
                            def JSA_2(df_JSA: pd.DataFrame) -> pd.DataFrame:
                                df_JSA2 = df_JSA[(df_JSA["IDD"].isin(IDD2))]
                                return df_JSA2.loc[1:, ["id","Chantier","NAct","NJSA","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_JSA1 = JSA_2(df_JSA)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()


                            df_JSA1['Date'] = pd.to_datetime(df_JSA1['Date']).apply(lambda x: x.date())
                            df_JSA1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_JSA1['Date']),key=4)
                                maxy= st.date_input('MaxDate',max(df_JSA1['Date']),key=4)

                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas au moins deux dates enregistrées.")
                                st.stop()

                            #filtrage par chantier
                            splitted_df_JSA1 = df_JSA1['Chantier'].str.split(',')
                            unique_vals6 = list(dict.fromkeys([y for x in splitted_df_JSA1  for y in x]).keys())
                            filtrechantier6 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals6,key=5)
                            
                            mask = (df_JSA1['Date'] > miny) & (df_JSA1['Date'] <= maxy) & (df_JSA1['Chantier'] == filtrechantier6)
                            df_filter6=df_JSA1.loc[mask]
                            st.dataframe(df_filter6)
                            st.text("*NAct: Nombre d'activité\n*NJSA: Analyse des risques réalisés")

                            if st.button("Télécharger", key=5):
                                st.markdown(get_table_download_link(df_filter6), unsafe_allow_html=True)
                            #figure
                            df_filter6['NAct'] = pd.to_numeric(df_filter6['NAct'])
                            df_filter6['NJSA'] = pd.to_numeric(df_filter6['NJSA'])
                            Objectf_fixé6= df_filter6['NAct'].sum()
                            Objectif_atteint6 = df_filter6['NJSA'].sum()
                            df_filter6_df = pd.DataFrame(columns=["NAct", "NJSA"])
                            df_filter6_df.at[0, "NAct"] = Objectf_fixé6
                            df_filter6_df.at[0, "NJSA"] = Objectif_atteint6
                            df_filter6_df_melt = pd.melt(df_filter6_df)
                            df_filter6_df_melt.columns = ['variable', 'valeur']

                            st.dataframe(df_filter6_df_melt)

                            figJSA = px.bar(df_filter6_df_melt, x = 'variable', y = 'valeur',color="variable")
                            st.plotly_chart(figJSA, use_container_width=True)



                        INCIDENT_exp= st.beta_expander("INCIDENT & ACCIDENT")
                        with INCIDENT_exp:

                            #IA
                            df_IA = pd.DataFrame(view_Incident_Accident(), columns=["id","IDD","Chantier","NInc","AAA","ASA","AT","NJP","Date"])
                            IDD2 = email.strip('][').split(', ')
			    
                            @st.cache
                            def IA_2(df_IA: pd.DataFrame) -> pd.DataFrame:
                                df_IA = df_IA[(df_IA["IDD"].isin(IDD2))]
                                return df_IA.loc[1:, ["id","Chantier","NInc","AAA","ASA","AT","NJP","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_IA1 = IA_2(df_IA)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_IA1['Date'] = pd.to_datetime(df_IA1['Date']).apply(lambda x: x.date())
                            df_IA1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_IA1['Date']),key=5)
                                maxy= st.date_input('MaxDate',max(df_IA1['Date']),key=5)
                            except:
                                st.error("Nous ne pouvons afficher car vous n'avez pas aumoins deux dates enrégistrées")
                                st.stop()


                            #filtrage par chantier
                            splitted_df_IA1 = df_IA1['Chantier'].str.split(',')
                            unique_vals7 = list(dict.fromkeys([y for x in splitted_df_IA1  for y in x]).keys())
                            filtrechantier7 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals7,key=6)
                            
                            mask = (df_IA1['Date'] > miny) & (df_IA1['Date'] <= maxy) & (df_IA1['Chantier'] == filtrechantier7)
                            df_filter7=df_IA1.loc[mask]
                            st.dataframe(df_filter7)
                            st.text("*NInc: Incident\n*AAA: Accident avec arrêt\n*ASA: Accident sans arrêt\n*AT:Accident de trajet\n*NJP:Nombre de jours perdus")

                            if st.button("Télécharger", key=6):
                                st.markdown(get_table_download_link(df_filter7), unsafe_allow_html=True)
                            #figure
                            df_filter7['NInc'] = pd.to_numeric(df_filter7['NInc'])
                            df_filter7['AAA'] = pd.to_numeric(df_filter7['AAA'])
                            df_filter7['ASA'] = pd.to_numeric(df_filter7['ASA'])
                            df_filter7['AT'] = pd.to_numeric(df_filter7['AT'])
                            df_filter7['NJP'] = pd.to_numeric(df_filter7['NJP'])



                            Objectf_fixe6 = df_filter7['NInc'].sum()
                            Objectf_fixe7 = df_filter7['AAA'].sum()
                            Objectf_fixe8= df_filter7['ASA'].sum()
                            Objectf_fixe9= df_filter7['AT'].sum()
                            Objectf_fixe10 = df_filter7['NJP'].sum()


                            df_filter7_df1 = pd.DataFrame(columns=["NInc","AAA","ASA","AT","NJP"])


                            df_filter7_df1.at[0, "NInc"] = Objectf_fixe6
                            df_filter7_df1.at[0, "AAA"] = Objectf_fixe7
                            df_filter7_df1.at[0, "ASA"] = Objectf_fixe8
                            df_filter7_df1.at[0, "AT"] = Objectf_fixe9
                            df_filter7_df1.at[0, "NJP"] = Objectf_fixe10


                            df_filter7_df_melt1 = pd.melt(df_filter7_df1)


                            df_filter7_df_melt1.columns = ['variable', 'valeur']


                            st.dataframe(df_filter7_df_melt1)


                            figIA = px.bar(df_filter7_df_melt1, x = 'variable', y = 'valeur',color="variable")


                            st.plotly_chart(figIA, use_container_width=True)





                        AUDIT_exp= st.beta_expander("AUDIT CHANTIER; VISITE CONJOINTE;  PRÉVENTION ET INSPECTION")
                        with AUDIT_exp:
                            #Audit
                            df_Audit = pd.DataFrame(view_Audit(), columns=["id","IDD","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"])
                            IDD2 = email.strip('][').split(', ')
			    
                            @st.cache
                            def Audit_2(df_Audit: pd.DataFrame) -> pd.DataFrame:
                                df_Audit = df_Audit[(df_Audit["IDD"].isin(IDD2))]
                                return df_Audit.loc[1:, ["id","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"]]

                            # Pour empêcher l'affichage d'erreur en cas de donnée vide
                            try:
                                df_Audit1 = Audit_2(df_Audit)
                            except:
                                st.error("Nous ne pouvons afficher, car vous n'avez pas de donnée enregistrée.")
                                st.stop()

                            df_Audit1['Date'] = pd.to_datetime(df_Audit1['Date']).apply(lambda x: x.date())
                            df_Audit1.sort_values(by=['Date'], inplace=True)

                            #intervalle de date
                            st.write('SELECTIONNEZ UN INTERVALLE DE DATE POUR VOTRE GRILLE')
                            try:
                                miny= st.date_input('MinDate',min(df_Audit1['Date']),key=6)
                                maxy= st.date_input('MaxDate',max(df_Audit1['Date']),key=6)
                            except:
                                st.error("Nous ne pouvons afficher car vous n'avez pas aumoins deux dates enrégistrées")
                                st.stop()


                            #filtrage par chantier
                            splitted_df_Audit1 = df_Audit1['Chantier'].str.split(',')
                            unique_vals8 = list(dict.fromkeys([y for x in splitted_df_Audit1  for y in x]).keys())
                            filtrechantier8 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals8,key=7)
                            
                            mask = (df_Audit1['Date'] > miny) & (df_Audit1['Date'] <= maxy) & (df_Audit1['Chantier'] == filtrechantier8)
                            df_filter8=df_Audit1.loc[mask]
                            st.dataframe(df_filter8)
                            st.text("*AC: Audit Chantier\n*VC:Visite conjointe\n*NEU:Nombre d'exercice d'urgence\n*SMPAR:Sensibilisation au modes de prévention des activités à risques\n*NPR:Nombre de procedures réalisées\n*IE:Inspections Environne-mentales")

                            if st.button("Télécharger", key=7):
                                st.markdown(get_table_download_link(df_filter8), unsafe_allow_html=True)
                            #figure
                            df_filter8['AC'] = pd.to_numeric(df_filter8['AC'])
                            df_filter8['VC'] = pd.to_numeric(df_filter8['VC'])
                            df_filter8['NEU'] = pd.to_numeric(df_filter8['NEU'])
                            df_filter8['SMPAR'] = pd.to_numeric(df_filter8['SMPAR'])
                            df_filter8['NPR'] = pd.to_numeric(df_filter8['NPR'])
                            df_filter8['IE'] = pd.to_numeric(df_filter8['IE'])

                            Objectf_fixe12 = df_filter8['AC'].sum()
                            Objectf_fixe13 = df_filter8['VC'].sum()
                            Objectf_fixe14= df_filter8['NEU'].sum()
                            Objectf_fixe15= df_filter8['SMPAR'].sum()
                            Objectf_fixe16 = df_filter8['NPR'].sum()
                            Objectf_fixe17 = df_filter8['IE'].sum()

                            df_filter8_df1 = pd.DataFrame(columns=["AC", "VC","NEU","SMPAR","NPR","IE"])


                            df_filter8_df1.at[0, "AC"] = Objectf_fixe12
                            df_filter8_df1.at[0, "VC"] = Objectf_fixe13
                            df_filter8_df1.at[0, "NEU"] = Objectf_fixe14
                            df_filter8_df1.at[0, "SMPAR"] = Objectf_fixe15
                            df_filter8_df1.at[0, "NPR"] = Objectf_fixe16
                            df_filter8_df1.at[0, "IE"] = Objectf_fixe17

                            df_filter8_df_melt1 = pd.melt(df_filter8_df1)


                            df_filter8_df_melt1.columns = ['variable', 'valeur']

                            st.dataframe(df_filter8_df_melt1)
                            figAC1 = px.bar(df_filter8_df_melt1, x = 'variable', y = 'valeur',color="variable")
                            st.plotly_chart(figAC1, use_container_width=True)

                    #Modification
                    elif choix == "METTRE À JOUR":
                        st.subheader("MODIFIER DES DONNÉES")
                        with st.beta_expander("ACCUEIL SECURITÉ"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Accueil = pd.DataFrame(view_Accueil(), columns=["id","IDD","Chantier","Nbre_Arrivant","Nbre_induction","Date"])

                            #pour voir uniquement les donnée de l'user connecté
                            IDD2 = email.strip('][').split(', ')

                            #ACCUEIL

                            @st.cache
                            def Accueil_2(df_Accueil: pd.DataFrame) -> pd.DataFrame:
                                df_Accueil2 = df_Accueil[(df_Accueil["IDD"].isin(IDD2))]
                                return df_Accueil2.loc[1:, ["id","Chantier","Nbre_Arrivant","Nbre_induction","Date"]]

                            df_Accueil1 = Accueil_2(df_Accueil)
                            
                            #filtrage par chantier
                            splitted_df_Accueil1 = df_Accueil1['Chantier'].str.split(',')
                            unique_vals1 = list(dict.fromkeys([y for x in splitted_df_Accueil1  for y in x]).keys())
                            filtrechantier = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals1,key=9)
                            mask =  (df_Accueil1['Chantier'] == filtrechantier)
                            df_filter1=df_Accueil1.loc[mask]
                            st.dataframe(df_filter1)


                            
                            idval = list(df_filter1['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval)
                            name_result = get_id_Accueil(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NArrivant = name_result[0][3]
                                Ninduction = name_result[0][4]
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NArrivant =inputcheck(st.text_input("Nombre Arrivant",NArrivant))
                                    new_Ninduction = inputcheck(st.text_input("Nombre d'induction",Ninduction))
                                    id=selected_id
                                    
        
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS")
                                if button1:
                                    edit_Accueil(new_Chantier,new_NArrivant,new_Ninduction,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')

                                df_Accueil = pd.DataFrame(view_Accueil(), columns=["id","IDD","Chantier","Nbre_Arrivant","Nbre_induction","Date"])
                                
                                #pour voir uniquement les donnée de l'user connecté
                                IDD2 = email.strip('][').split(', ')
        
                                #ACCUEIL
        
                                @st.cache
                                def Accueil_2(df_Accueil: pd.DataFrame) -> pd.DataFrame:
                                    df_Accueil2 = df_Accueil[(df_Accueil["IDD"].isin(IDD2))]
                                    return df_Accueil2.loc[1:, ["id","Chantier","Nbre_Arrivant","Nbre_induction","Date"]]
        
                                df_Accueil1 = Accueil_2(df_Accueil)
                                #filtrage par chantier
                                splitted_df_Accueil1 = df_Accueil1['Chantier'].str.split(',')
                                unique_vals1 = list(dict.fromkeys([y for x in splitted_df_Accueil1  for y in x]).keys())
                                filtrechantier = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals1,key=10)
                                mask =  (df_Accueil1['Chantier'] == filtrechantier)
                                df_filter1=df_Accueil1.loc[mask]
                                st.dataframe(df_filter1)
                                    
                        
                        
                        
                        with st.beta_expander("BRIEFING DE SÉCURITÉ( TBM)"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_TBM = pd.DataFrame(view_TBM(), columns=["id","IDD","Chantier","Nbre_chantier","Nbre_TBM","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def TBM_2(df_TBM: pd.DataFrame) -> pd.DataFrame:
                                df_TBM2 = df_TBM[(df_TBM["IDD"].isin(IDD2))]
                                return df_TBM2.loc[1:, ["id","Chantier","Nbre_chantier","Nbre_TBM","Date"]]

                            df_TBM1 = TBM_2(df_TBM)
                            #filtrage par chantier
                            splitted_df_TBM1 = df_TBM1['Chantier'].str.split(',')
                            unique_vals2 = list(dict.fromkeys([y for x in splitted_df_TBM1  for y in x]).keys())
                            filtrechantier2 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals2,key=100)
                            
                            mask =  (df_TBM1['Chantier'] == filtrechantier2)
                            df_filter2=df_TBM1.loc[mask]
                            st.dataframe(df_filter2)
                            


                            
                            idval = list(df_filter2['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=9)
                            name_result = get_id_TBM(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NChantier = name_result[0][3]
                                NTBM = name_result[0][4]
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NChantier =inputcheck(st.text_input("Nombre Arrivant",NChantier,key=0))
                                    new_NTBM = inputcheck(st.text_input("Nombre d'induction",NTBM,key=1))
                                    id=selected_id
                                    
        
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier,key=2)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=0)
                                if button1:
                                     edit_TBM(new_Chantier,new_NChantier,new_NTBM,id)
                                     st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                     
                                st.markdown('### DONNÉE MODIFIÉE')

                                df_TBM = pd.DataFrame(view_TBM(), columns=["id","IDD","Chantier","Nbre_chantier","Nbre_TBM","Date"])
                                IDD2 = email.strip('][').split(', ')
                                @st.cache
                                def TBM_2(df_TBM: pd.DataFrame) -> pd.DataFrame:
                                    df_TBM2 = df_TBM[(df_TBM["IDD"].isin(IDD2))]
                                    return df_TBM2.loc[1:, ["id","Chantier","Nbre_chantier","Nbre_TBM","Date"]]

                                df_TBM1 = TBM_2(df_TBM)
                                #filtrage par chantier
                                splitted_df_TBM1 = df_TBM1['Chantier'].str.split(',')
                                unique_vals2 = list(dict.fromkeys([y for x in splitted_df_TBM1  for y in x]).keys())
                                filtrechantier2 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals2,key=101)
                                mask =  (df_TBM1['Chantier'] == filtrechantier2)
                                df_filter2=df_TBM1.loc[mask]
                                st.dataframe(df_filter2)
                                
                        
                        with st.beta_expander("NON CONFORMITÉ"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_NC = pd.DataFrame(view_NC(), columns=["id","IDD","Chantier","NCR","FNCR","NCC","FNCC","Date"])
                            IDD2 = email.strip('][').split(', ')


                            @st.cache
                            def NC_2(df_NC: pd.DataFrame) -> pd.DataFrame:
                                df_NC2 = df_NC[(df_NC["IDD"].isin(IDD2))]
                                return df_NC2.loc[1:, ["id","Chantier","NCR","FNCR","NCC","FNCC","Date"]]

                            df_NC1 = NC_2(df_NC)

                            #filtrage par chantier
                            splitted_df_NC1 = df_NC1['Chantier'].str.split(',')
                            unique_vals3 = list(dict.fromkeys([y for x in splitted_df_NC1  for y in x]).keys())
                            filtrechantier3 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals3,key=21)
                            mask =  (df_NC1['Chantier'] == filtrechantier3)
                            df_filter3=df_NC1.loc[mask]
                            st.dataframe(df_filter3)
                            


                            
                            idval = list(df_filter3['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=0)
                            name_result = get_id_NC(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NCR = name_result[0][3]
                                FNCR = name_result[0][4]
                                NCC = name_result[0][5]
                                FNCC = name_result[0][6]
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NCR = inputcheck(st.text_input("Nombre de Non conformité remontée",NCR,key=0))
                                    new_FNCR = inputcheck(st.text_input("Nombre de fiche de Non conformité remontée",FNCR,key=1))
                                    new_NCC = inputcheck(st.text_input("Nombre de Non conformité cloturée",NCC,key=2))
                                    new_FNCC= inputcheck(st.text_input("Nombre de fiche de Non conformité cloturée",FNCC, key=3))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier,key=4)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=1)
                                if button1:
                                    edit_NC(new_Chantier,new_NCR,new_FNCR,new_NCC,new_FNCC,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_NC = pd.DataFrame(view_NC(), columns=["id","IDD","Chantier","NCR","FNCR","NCC","FNCC","Date"])
                                IDD2 = email.strip('][').split(', ')


                                @st.cache
                                def NC_2(df_NC: pd.DataFrame) -> pd.DataFrame:
                                    df_NC2 = df_NC[(df_NC["IDD"].isin(IDD2))]
                                    return df_NC2.loc[1:, ["id","Chantier","NCR","FNCR","NCC","FNCC","Date"]]

                                df_NC1 = NC_2(df_NC)
                                #filtrage par chantier
                                splitted_df_NC1 = df_NC1['Chantier'].str.split(',')
                                unique_vals3 = list(dict.fromkeys([y for x in splitted_df_NC1  for y in x]).keys())
                                filtrechantier3 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals3,key=22)
                                mask =  (df_NC1['Chantier'] == filtrechantier3)
                                df_filter3=df_NC1.loc[mask]
                                st.dataframe(df_filter3)


                        with st.beta_expander("CHANGEMENTS ENREGISTRÉS"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Changements = pd.DataFrame(view_Changements(), columns=["id","IDD","Chantier","NCH","FNCH","NCHC","FNCHC","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Changements_2(df_Changements: pd.DataFrame) -> pd.DataFrame:
                                df_Changements2 = df_Changements[(df_Changements["IDD"].isin(IDD2))]
                                return df_Changements2.loc[1:, ["id","Chantier","NCH","FNCH","NCHC","FNCHC","Date"]]

                            df_Changements1 = Changements_2(df_Changements)
                            #filtrage par chantier
                            splitted_df_Changements1 = df_Changements1['Chantier'].str.split(',')
                            unique_vals4 = list(dict.fromkeys([y for x in splitted_df_Changements1  for y in x]).keys())
                            filtrechantier4 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals4,key=12)
                            mask = (df_Changements1['Chantier'] == filtrechantier4)
                            df_filter4=df_Changements1.loc[mask]
                            st.dataframe(df_filter4)
                            

                            idval = list(df_filter4['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=1)
                            name_result = get_id_Changements(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NCH = name_result[0][3]
                                FNCH = name_result[0][4]
                                NCHC = name_result[0][5]
                                FNCHC = name_result[0][6]
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NCH = inputcheck(st.text_input("Nombre de Changement enregistrés",NCH))
                                    new_FNCH = inputcheck(st.text_input("Nombre de fiche de Changements enregistrés",FNCH))
                                    new_NCHC  = inputcheck(st.text_input("Nombre de Changements cloturés",NCHC))
                                    new_FNCHC= inputcheck(st.text_input("Nombre de fiche de  Changements suivis et cloturés",FNCHC))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    Chantier = st.text_input("Chantier",Chantier,key=3)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=3)
                                if button1:
                                    edit_Changements(new_Chantier,new_NCH,new_FNCH,new_NCHC,new_FNCHC,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Changements = pd.DataFrame(view_Changements(), columns=["id","IDD","Chantier","NCH","FNCH","NCHC","FNCHC","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Changements_2(df_Changements: pd.DataFrame) -> pd.DataFrame:
                                    df_Changements2 = df_Changements[(df_Changements["IDD"].isin(IDD2))]
                                    return df_Changements2.loc[1:, ["id","Chantier","NCH","FNCH","NCHC","FNCHC","Date"]]

                                df_Changements1 = Changements_2(df_Changements)
                                #filtrage par chantier
                                splitted_df_Changements1 = df_Changements1['Chantier'].str.split(',')
                                unique_vals4 = list(dict.fromkeys([y for x in splitted_df_Changements1  for y in x]).keys())
                                filtrechantier4 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals4,key=13)
                                mask = (df_Changements1['Chantier'] == filtrechantier4)
                                df_filter4=df_Changements1.loc[mask]
                                st.dataframe(df_filter4)


                        with st.beta_expander("ANOMALIES"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Anomalies = pd.DataFrame(view_Anomalies(), columns=["id","IDD","Chantier","NA","FNA","NAC","FNAC","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Anomalies_2(df_Anomalies: pd.DataFrame) -> pd.DataFrame:
                                df_Anomalies2 = df_Anomalies[(df_Anomalies["IDD"].isin(IDD2))]
                                return df_Anomalies2.loc[1:, ["id","Chantier","NA","FNA","NAC","FNAC","Date"]]

                            df_Anomalies1 = Anomalies_2(df_Anomalies)
                            #filtrage par chantier
                            splitted_df_Anomalies1 = df_Anomalies1['Chantier'].str.split(',')
                            unique_vals5 = list(dict.fromkeys([y for x in splitted_df_Anomalies1  for y in x]).keys())
                            filtrechantier5 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals5,key=14)
                            mask = (df_Anomalies1['Chantier'] == filtrechantier5)
                            df_filter5=df_Anomalies1.loc[mask]
                            st.dataframe(df_filter5)
                            

                            idval = list(df_filter5['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE A MODIFIER", idval,key=4)
                            name_result = get_id_Anomalies(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NA = name_result[0][3]
                                FNA = name_result[0][4]
                                NAC = name_result[0][5]
                                FNAC = name_result[0][6]
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NA = inputcheck(st.text_input("Nombre d'Anomalies Remontées",NA))
                                    new_FNA = inputcheck(st.text_input("Nombre de fiche d'Anomalies Remontées",FNA))
                                    new_NAC = inputcheck(st.text_input("Nombre d' Anomalies cloturés",NAC))
                                    new_FNAC = inputcheck(st.text_input("Nombre de fiche de  Anomalies Corrigées",FNAC))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier,key=5)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=5)
                                if button1:
                                    edit_Anomalies(new_Chantier,new_NA,new_FNA,new_NAC,new_FNAC,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Anomalies = pd.DataFrame(view_Anomalies(), columns=["id","IDD","Chantier","NA","FNA","NAC","FNAC","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Anomalies_2(df_Anomalies: pd.DataFrame) -> pd.DataFrame:
                                    df_Anomalies2 = df_Anomalies[(df_Anomalies["IDD"].isin(IDD2))]
                                    return df_Anomalies2.loc[1:, ["id","Chantier","NA","FNA","NAC","FNAC","Date"]]

                                df_Anomalies1 = Anomalies_2(df_Anomalies)
                                #filtrage par chantier
                                splitted_df_Anomalies1 = df_Anomalies1['Chantier'].str.split(',')
                                unique_vals5 = list(dict.fromkeys([y for x in splitted_df_Anomalies1  for y in x]).keys())
                                filtrechantier5 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals5,key=144)
                                
                                df_filter5=df_Anomalies1.loc[mask]
                                st.dataframe(df_filter5)

                        
                        with st.beta_expander("ANALYSE DES RISQUES RÉALISÉS(JSA)"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_JSA = pd.DataFrame(view_JSA(), columns=["id","IDD","Chantier","NAct","NJSA","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def JSA_2(df_JSA: pd.DataFrame) -> pd.DataFrame:
                                df_JSA2 = df_JSA[(df_JSA["IDD"].isin(IDD2))]
                                return df_JSA2.loc[1:, ["id","Chantier","NAct","NJSA","Date"]]

                            df_JSA1 = JSA_2(df_JSA)
                            #filtrage par chantier
                            splitted_df_JSA1 = df_JSA1['Chantier'].str.split(',')
                            unique_vals6 = list(dict.fromkeys([y for x in splitted_df_JSA1  for y in x]).keys())
                            filtrechantier6 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals6,key=166)
                            mask = (df_JSA1['Chantier'] == filtrechantier6)
                            df_filter6=df_JSA1.loc[mask]
                            st.dataframe(df_filter6)

                            idval = list(df_filter6['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=6)
                            name_result = get_id_JSA(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NAct = name_result[0][3]
                                NJSA = name_result[0][4]
                               
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_NAct = inputcheck(st.text_input("Nombre d'Activite",NAct))
                                    new_NJSA = inputcheck(st.text_input("Nombre de fiche JSA",NJSA))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier,key=6)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=6)
                                if button1:
                                    edit_JSA(new_Chantier,new_NAct,new_NJSA,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_JSA = pd.DataFrame(view_JSA(), columns=["id","IDD","Chantier","NAct","NJSA","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def JSA_2(df_JSA: pd.DataFrame) -> pd.DataFrame:
                                    df_JSA2 = df_JSA[(df_JSA["IDD"].isin(IDD2))]
                                    return df_JSA2.loc[1:, ["id","Chantier","NAct","NJSA","Date"]]

                                df_JSA1 = JSA_2(df_JSA)
                                #filtrage par chantier
                                splitted_df_JSA1 = df_JSA1['Chantier'].str.split(',')
                                unique_vals6 = list(dict.fromkeys([y for x in splitted_df_JSA1  for y in x]).keys())
                                filtrechantier6 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals6,key=177)
                                mask = (df_JSA1['Chantier'] == filtrechantier6)
                                df_filter6=df_JSA1.loc[mask]
                                st.dataframe(df_filter6)



                        with st.beta_expander("INCIDENT & ACCIDENT"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_IA = pd.DataFrame(view_Incident_Accident(), columns=["id","IDD","Chantier","NInc","AAA","ASA","AT","NJP","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def IA_2(df_IA: pd.DataFrame) -> pd.DataFrame:
                                df_IA = df_IA[(df_IA["IDD"].isin(IDD2))]
                                return df_IA.loc[1:, ["id","Chantier","NInc","AAA","ASA","AT","NJP","Date"]]

                            df_IA1 = IA_2(df_IA)

                            #filtrage par chantier
                            splitted_df_IA1 = df_IA1['Chantier'].str.split(',')
                            unique_vals7 = list(dict.fromkeys([y for x in splitted_df_IA1  for y in x]).keys())
                            filtrechantier7 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals7,key=18)
                            mask = (df_IA1['Chantier'] == filtrechantier7)
                            df_filter7=df_IA1.loc[mask]
                            st.dataframe(df_filter7)

                            idval = list(df_filter7['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=7)
                            name_result = get_id_Incident_Accident(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                NInc = name_result[0][3]
                                AAA = name_result[0][4]
                                ASA = name_result[0][5]
                                AT = name_result[0][6]
                                NJP = name_result[0][7]
                               
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_AAA = inputcheck(st.text_input("Accident Avec Arrêt",AAA))
                                    new_NJP = inputcheck(st.text_input("Nombre de jours perdus",NJP))
                                    new_ASA = inputcheck(st.text_input("Accident Sans Arrêt",ASA))
                                    new_AT = inputcheck(st.text_input("Nombre d'accident de trajet",AT))
                                    new_NInc = inputcheck(st.text_input("Incident",NInc))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier",Chantier,key=7)
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=7)
                                if button1:
                                    edit_Incident_Accident(new_Chantier,new_NInc,new_AAA,new_ASA,new_AT,new_NJP,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))

                                df_IA = pd.DataFrame(view_Incident_Accident(), columns=["id","IDD","Chantier","NInc","AAA","ASA","AT","NJP","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def IA_2(df_IA: pd.DataFrame) -> pd.DataFrame:
                                    df_IA = df_IA[(df_IA["IDD"].isin(IDD2))]
                                    return df_IA.loc[1:, ["id","Chantier","NInc","AAA","ASA","AT","NJP","Date"]]

                                df_IA1 = IA_2(df_IA)
                                splitted_df_IA1 = df_IA1['Chantier'].str.split(',')
                                unique_vals7 = list(dict.fromkeys([y for x in splitted_df_IA1  for y in x]).keys())
                                filtrechantier7 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals7,key=19)
                                mask = (df_IA1['Chantier'] == filtrechantier7)
                                df_filter7=df_IA1.loc[mask]
                                st.dataframe(df_filter7)




                        with st.beta_expander("AUDIT CHANTIER; VISITE CONJOINTE;  PRÉVENTION ET INSPECTION"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Audit = pd.DataFrame(view_Audit(), columns=["id","IDD","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Audit_2(df_Audit: pd.DataFrame) -> pd.DataFrame:
                                df_Audit = df_Audit[(df_Audit["IDD"].isin(IDD2))]
                                return df_Audit.loc[1:, ["id","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"]]

                            df_Audit1 = Audit_2(df_Audit)
                            #filtrage par chantier
                            splitted_df_Audit1 = df_Audit1['Chantier'].str.split(',')
                            unique_vals8 = list(dict.fromkeys([y for x in splitted_df_Audit1  for y in x]).keys())
                            filtrechantier8 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals8,key=20)
                            mask = (df_Audit1['Chantier'] == filtrechantier8)
                            df_filter8=df_Audit1.loc[mask]
                            st.dataframe(df_filter8)

                            idval = list(df_filter8['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À MODIFIER", idval,key=8)
                            name_result = get_id_Audit(selected_id)

                            if name_result:
                                id = name_result[0][0]
                                Chantier = name_result[0][2]
                                AC = name_result[0][3]
                                VC = name_result[0][4]
                                NEU = name_result[0][5]
                                SMPAR = name_result[0][6]
                                NPR = name_result[0][7]
                                IE = name_result[0][8]
                               
                                
                                col1, col2= st.beta_columns(2)
                                with col1:
                                    st.subheader("CIBLE À MODIFIER")
                                with col1:
                                    new_AC= inputcheck(st.text_input("Nombre d'audit",AC))
                                    new_VC= inputcheck(st.text_input("Nombre de Visite Conjointe",VC))
                                    new_NEU= inputcheck(st.text_input("Nombre d'exercice d'urgence",NEU))
                                    new_SMPAR= inputcheck(st.text_input("Sensibilisation au modes de prévention des activités à risques",SMPAR))
                                    new_NPR= inputcheck(st.text_input("Procedures réalisées",NPR))
                                    new_IE= inputcheck(st.text_input("Inspections Environnementales",IE))
                                    
                                with col2:
                                    st.subheader("NOM DU CHANTIER")
                                    new_Chantier = st.text_input("Chantier")
                                    
                                button1=st.button("MODIFIER LES DÉTAILS",key=8)
                                if button1:
                                    edit_Audit(new_ID,new_Chantier,new_AC,new_VC,new_NEU,new_SMPAR,new_NPR,new_IE,id)
                                    st.success("MODIFIÉ AVEC SUCCÈS: {}".format(new_Chantier))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Audit = pd.DataFrame(view_Audit(), columns=["id","IDD","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Audit_2(df_Audit: pd.DataFrame) -> pd.DataFrame:
                                    df_Audit = df_Audit[(df_Audit["IDD"].isin(IDD2))]
                                    return df_Audit.loc[1:, ["id","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"]]

                                df_Audit1 = Audit_2(df_Audit)
                                #filtrage par chantier
                                splitted_df_Audit1 = df_Audit1['Chantier'].str.split(',')
                                unique_vals8 = list(dict.fromkeys([y for x in splitted_df_Audit1  for y in x]).keys())
                                filtrechantier8 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals8,key=211)
                                mask = (df_Audit1['Chantier'] == filtrechantier8)
                                df_filter8=df_Audit1.loc[mask]
                                st.dataframe(df_filter8)

                    #Suppression des données
                    elif choix ==  "SUPPRIMER":
                        st.subheader("SUPPRIMER DES DONNÉES")
                        with st.beta_expander("ACCUEIL SECURITÉ"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Accueil = pd.DataFrame(view_Accueil(), columns=["id","IDD","Chantier","Nbre_Arrivant","Nbre_induction","Date"])

                            #pour voir uniquement les donnée de l'user connecté
                            IDD2 = email.strip('][').split(', ')

                            #ACCUEIL

                            @st.cache
                            def Accueil_2(df_Accueil: pd.DataFrame) -> pd.DataFrame:
                                df_Accueil2 = df_Accueil[(df_Accueil["IDD"].isin(IDD2))]
                                return df_Accueil2.loc[1:, ["id","Chantier","Nbre_Arrivant","Nbre_induction","Date"]]

                            df_Accueil1 = Accueil_2(df_Accueil)
                            
                            #filtrage par chantier
                            splitted_df_Accueil1 = df_Accueil1['Chantier'].str.split(',')
                            unique_vals1 = list(dict.fromkeys([y for x in splitted_df_Accueil1  for y in x]).keys())
                            filtrechantier = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals1,key=22)
                            mask =  (df_Accueil1['Chantier'] == filtrechantier)
                            df_filter1=df_Accueil1.loc[mask]
                            st.dataframe(df_filter1)

                            idval = list(df_filter1['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval, key=10)
                            name_delete = get_id_Accueil(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER"):
                                    delete_data_Accueil(id)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Accueil = pd.DataFrame(view_Accueil(), columns=["id","IDD","Chantier","Nbre_Arrivant","Nbre_induction","Date"])

                                #pour voir uniquement les donnée de l'user connecté
                                IDD2 = email.strip('][').split(', ')

                                #ACCUEIL

                                @st.cache
                                def Accueil_2(df_Accueil: pd.DataFrame) -> pd.DataFrame:
                                    df_Accueil2 = df_Accueil[(df_Accueil["IDD"].isin(IDD2))]
                                    return df_Accueil2.loc[1:, ["id","Chantier","Nbre_Arrivant","Nbre_induction","Date"]]

                                df_Accueil1 = Accueil_2(df_Accueil)
                                
                                #filtrage par chantier
                                splitted_df_Accueil1 = df_Accueil1['Chantier'].str.split(',')
                                unique_vals1 = list(dict.fromkeys([y for x in splitted_df_Accueil1  for y in x]).keys())
                                filtrechantier = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals1,key=23)
                                mask =  (df_Accueil1['Chantier'] == filtrechantier)
                                df_filter1=df_Accueil1.loc[mask]
                                st.dataframe(df_filter1)



                        with st.beta_expander("BRIEFING DE SÉCURITÉ( TBM)"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_TBM = pd.DataFrame(view_TBM(), columns=["id","IDD","Chantier","Nbre_chantier","Nbre_TBM","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def TBM_2(df_TBM: pd.DataFrame) -> pd.DataFrame:
                                df_TBM2 = df_TBM[(df_TBM["IDD"].isin(IDD2))]
                                return df_TBM2.loc[1:, ["id","Chantier","Nbre_chantier","Nbre_TBM","Date"]]

                            df_TBM1 = TBM_2(df_TBM)
                            #filtrage par chantier
                            splitted_df_TBM1 = df_TBM1['Chantier'].str.split(',')
                            unique_vals2 = list(dict.fromkeys([y for x in splitted_df_TBM1  for y in x]).keys())
                            filtrechantier2 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals2,key=24)
                            mask =  (df_TBM1['Chantier'] == filtrechantier2)
                            df_filter2=df_TBM1.loc[mask]
                            st.dataframe(df_filter2)

                            
                            idval = list(df_filter2['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=41)
                            name_delete = get_id_TBM(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=0):
                                    delete_data_TBM(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_TBM = pd.DataFrame(view_TBM(), columns=["id","IDD","Chantier","Nbre_chantier","Nbre_TBM","Date"])
                                IDD2 = email.strip('][').split(', ')
                                @st.cache
                                def TBM_2(df_TBM: pd.DataFrame) -> pd.DataFrame:
                                    df_TBM2 = df_TBM[(df_TBM["IDD"].isin(IDD2))]
                                    return df_TBM2.loc[1:, ["id","Chantier","Nbre_chantier","Nbre_TBM","Date"]]

                                df_TBM1 = TBM_2(df_TBM)
                                #filtrage par chantier
                                splitted_df_TBM1 = df_TBM1['Chantier'].str.split(',')
                                unique_vals2 = list(dict.fromkeys([y for x in splitted_df_TBM1  for y in x]).keys())
                                filtrechantier2 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals2,key=35)
                                mask =  (df_TBM1['Chantier'] == filtrechantier2)
                                df_filter2=df_TBM1.loc[mask]
                                st.dataframe(df_filter2)


                        with st.beta_expander("NON CONFORMITÉ"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_NC = pd.DataFrame(view_NC(), columns=["id","IDD","Chantier","NCR","FNCR","NCC","FNCC","Date"])
                            IDD2 = email.strip('][').split(', ')


                            @st.cache
                            def NC_2(df_NC: pd.DataFrame) -> pd.DataFrame:
                                df_NC2 = df_NC[(df_NC["IDD"].isin(IDD2))]
                                return df_NC2.loc[1:, ["id","Chantier","NCR","FNCR","NCC","FNCC","Date"]]

                            df_NC1 = NC_2(df_NC)

                            #filtrage par chantier
                            splitted_df_NC1 = df_NC1['Chantier'].str.split(',')
                            unique_vals3 = list(dict.fromkeys([y for x in splitted_df_NC1  for y in x]).keys())
                            filtrechantier3 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals3,key=25)
                            mask =  (df_NC1['Chantier'] == filtrechantier3)
                            df_filter3=df_NC1.loc[mask]
                            st.dataframe(df_filter3)
                            
                            
                            idval = list(df_filter3['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=12)
                            name_delete = get_id_NC(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=1):
                                    delete_data_NC(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_NC = pd.DataFrame(view_NC(), columns=["id","IDD","Chantier","NCR","FNCR","NCC","FNCC","Date"])
                                IDD2 = email.strip('][').split(', ')


                                @st.cache
                                def NC_2(df_NC: pd.DataFrame) -> pd.DataFrame:
                                    df_NC2 = df_NC[(df_NC["IDD"].isin(IDD2))]
                                    return df_NC2.loc[1:, ["id","Chantier","NCR","FNCR","NCC","FNCC","Date"]]

                                df_NC1 = NC_2(df_NC)

                                #filtrage par chantier
                                splitted_df_NC1 = df_NC1['Chantier'].str.split(',')
                                unique_vals3 = list(dict.fromkeys([y for x in splitted_df_NC1  for y in x]).keys())
                                filtrechantier3 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals3,key=26)
                                mask =  (df_NC1['Chantier'] == filtrechantier3)
                                df_filter3=df_NC1.loc[mask]
                                st.dataframe(df_filter3)

                        with st.beta_expander("CHANGEMENTS ENREGISTRÉS"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Changements = pd.DataFrame(view_Changements(), columns=["id","IDD","Chantier","NCH","FNCH","NCHC","FNCHC","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Changements_2(df_Changements: pd.DataFrame) -> pd.DataFrame:
                                df_Changements2 = df_Changements[(df_Changements["IDD"].isin(IDD2))]
                                return df_Changements2.loc[1:, ["id","Chantier","NCH","FNCH","NCHC","FNCHC","Date"]]

                            df_Changements1 = Changements_2(df_Changements)
                            #filtrage par chantier
                            splitted_df_Changements1 = df_Changements1['Chantier'].str.split(',')
                            unique_vals4 = list(dict.fromkeys([y for x in splitted_df_Changements1  for y in x]).keys())
                            filtrechantier4 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals4,key=27)
                            mask = (df_Changements1['Chantier'] == filtrechantier4)
                            df_filter4=df_Changements1.loc[mask]
                            st.dataframe(df_filter4)
                            

                            idval = list(df_filter4['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=13)
                            name_delete = get_id_Changements(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=2):
                                    delete_data_Changements(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Changements = pd.DataFrame(view_Changements(), columns=["id","IDD","Chantier","NCH","FNCH","NCHC","FNCHC","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Changements_2(df_Changements: pd.DataFrame) -> pd.DataFrame:
                                    df_Changements2 = df_Changements[(df_Changements["IDD"].isin(IDD2))]
                                    return df_Changements2.loc[1:, ["id","Chantier","NCH","FNCH","NCHC","FNCHC","Date"]]

                                df_Changements1 = Changements_2(df_Changements)
                                #filtrage par chantier
                                splitted_df_Changements1 = df_Changements1['Chantier'].str.split(',')
                                unique_vals4 = list(dict.fromkeys([y for x in splitted_df_Changements1  for y in x]).keys())
                                filtrechantier4 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals4,key=28)
                                mask = (df_Changements1['Chantier'] == filtrechantier4)
                                df_filter4=df_Changements1.loc[mask]
                                st.dataframe(df_filter4)


                        with st.beta_expander("ANOMALIES"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Anomalies = pd.DataFrame(view_Anomalies(), columns=["id","IDD","Chantier","NA","FNA","NAC","FNAC","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Anomalies_2(df_Anomalies: pd.DataFrame) -> pd.DataFrame:
                                df_Anomalies2 = df_Anomalies[(df_Anomalies["IDD"].isin(IDD2))]
                                return df_Anomalies2.loc[1:, ["id","Chantier","NA","FNA","NAC","FNAC","Date"]]

                            df_Anomalies1 = Anomalies_2(df_Anomalies)
                            #filtrage par chantier
                            splitted_df_Anomalies1 = df_Anomalies1['Chantier'].str.split(',')
                            unique_vals5 = list(dict.fromkeys([y for x in splitted_df_Anomalies1  for y in x]).keys())
                            filtrechantier5 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals5,key=29)
                            mask = (df_Anomalies1['Chantier'] == filtrechantier5)
                            df_filter5=df_Anomalies1.loc[mask]
                            st.dataframe(df_filter5)
                            

                            idval = list(df_filter5['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=14)
                            name_delete = get_id_Anomalies(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=3):
                                    delete_data_Anomalies(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Anomalies = pd.DataFrame(view_Anomalies(), columns=["id","IDD","Chantier","NA","FNA","NAC","FNAC","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Anomalies_2(df_Anomalies: pd.DataFrame) -> pd.DataFrame:
                                    df_Anomalies2 = df_Anomalies[(df_Anomalies["IDD"].isin(IDD2))]
                                    return df_Anomalies2.loc[1:, ["id","Chantier","NA","FNA","NAC","FNAC","Date"]]

                                df_Anomalies1 = Anomalies_2(df_Anomalies)
                                #filtrage par chantier
                                splitted_df_Anomalies1 = df_Anomalies1['Chantier'].str.split(',')
                                unique_vals5 = list(dict.fromkeys([y for x in splitted_df_Anomalies1  for y in x]).keys())
                                filtrechantier5 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals5,key=30)
                                mask = (df_Anomalies1['Chantier'] == filtrechantier5)
                                df_filter5=df_Anomalies1.loc[mask]
                                st.dataframe(df_filter5)



                        with st.beta_expander("ANALYSE DES RISQUES RÉALISÉS(JSA)"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_JSA = pd.DataFrame(view_JSA(), columns=["id","IDD","Chantier","NAct","NJSA","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def JSA_2(df_JSA: pd.DataFrame) -> pd.DataFrame:
                                df_JSA2 = df_JSA[(df_JSA["IDD"].isin(IDD2))]
                                return df_JSA2.loc[1:, ["id","Chantier","NAct","NJSA","Date"]]

                            df_JSA1 = JSA_2(df_JSA)
                            #filtrage par chantier
                            splitted_df_JSA1 = df_JSA1['Chantier'].str.split(',')
                            unique_vals6 = list(dict.fromkeys([y for x in splitted_df_JSA1  for y in x]).keys())
                            filtrechantier6 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals6,key=31)
                            mask = (df_JSA1['Chantier'] == filtrechantier6)
                            df_filter6=df_JSA1.loc[mask]
                            st.dataframe(df_filter6)

                            idval = list(df_filter6['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=15)
                            name_delete = get_id_JSA(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=4):
                                    delete_data_JSA(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_JSA = pd.DataFrame(view_JSA(), columns=["id","IDD","Chantier","NAct","NJSA","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def JSA_2(df_JSA: pd.DataFrame) -> pd.DataFrame:
                                    df_JSA2 = df_JSA[(df_JSA["IDD"].isin(IDD2))]
                                    return df_JSA2.loc[1:, ["id","Chantier","NAct","NJSA","Date"]]

                                df_JSA1 = JSA_2(df_JSA)
                                #filtrage par chantier
                                splitted_df_JSA1 = df_JSA1['Chantier'].str.split(',')
                                unique_vals6 = list(dict.fromkeys([y for x in splitted_df_JSA1  for y in x]).keys())
                                filtrechantier6 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals6,key=32)
                                mask = (df_JSA1['Chantier'] == filtrechantier6)
                                df_filter6=df_JSA1.loc[mask]
                                st.dataframe(df_filter6)


                        with st.beta_expander("INCIDENT & ACCIDENT"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_IA = pd.DataFrame(view_Incident_Accident(), columns=["id","IDD","Chantier","NInc","AAA","ASA","AT","NJP","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def IA_2(df_IA: pd.DataFrame) -> pd.DataFrame:
                                df_IA = df_IA[(df_IA["IDD"].isin(IDD2))]
                                return df_IA.loc[1:, ["id","Chantier","NInc","AAA","ASA","AT","NJP","Date"]]

                            df_IA1 = IA_2(df_IA)

                            #filtrage par chantier
                            splitted_df_IA1 = df_IA1['Chantier'].str.split(',')
                            unique_vals7 = list(dict.fromkeys([y for x in splitted_df_IA1  for y in x]).keys())
                            filtrechantier7 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals7,key=33)
                            mask = (df_IA1['Chantier'] == filtrechantier7)
                            df_filter7=df_IA1.loc[mask]
                            st.dataframe(df_filter7)

                            idval = list(df_filter7['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=16)
                            name_delete = get_id_Incident_Accident(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=5):
                                    delete_data_Incident_Accident(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                    
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_IA = pd.DataFrame(view_Incident_Accident(), columns=["id","IDD","Chantier","NInc","AAA","ASA","AT","NJP","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def IA_2(df_IA: pd.DataFrame) -> pd.DataFrame:
                                    df_IA = df_IA[(df_IA["IDD"].isin(IDD2))]
                                    return df_IA.loc[1:, ["id","Chantier","NInc","AAA","ASA","AT","NJP","Date"]]

                                df_IA1 = IA_2(df_IA)

                                #filtrage par chantier
                                splitted_df_IA1 = df_IA1['Chantier'].str.split(',')
                                unique_vals7 = list(dict.fromkeys([y for x in splitted_df_IA1  for y in x]).keys())
                                filtrechantier7 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals7,key=34)
                                mask = (df_IA1['Chantier'] == filtrechantier7)
                                df_filter7=df_IA1.loc[mask]
                                st.dataframe(df_filter7)


                        with st.beta_expander("AUDIT CHANTIER; VISITE CONJOINTE;  PRÉVENTION ET INSPECTION"):
                            st.markdown('### DONNÉE ACTUELLE')
                            df_Audit = pd.DataFrame(view_Audit(), columns=["id","IDD","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"])
                            IDD2 = email.strip('][').split(', ')

                            @st.cache
                            def Audit_2(df_Audit: pd.DataFrame) -> pd.DataFrame:
                                df_Audit = df_Audit[(df_Audit["IDD"].isin(IDD2))]
                                return df_Audit.loc[1:, ["id","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"]]

                            df_Audit1 = Audit_2(df_Audit)
                            #filtrage par chantier
                            splitted_df_Audit1 = df_Audit1['Chantier'].str.split(',')
                            unique_vals8 = list(dict.fromkeys([y for x in splitted_df_Audit1  for y in x]).keys())
                            filtrechantier8 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals8,key=355)
                            mask = (df_Audit1['Chantier'] == filtrechantier8)
                            df_filter8=df_Audit1.loc[mask]
                            st.dataframe(df_filter8)

                            idval = list(df_filter8['id'])
                            selected_id = st.selectbox("SELECTIONEZ l'ID DE LA LIGNE À SUPPRIMER", idval,key=17)
                            name_delete = get_id_Audit(selected_id)
                            if name_delete:
                                id = name_delete[0][0]
                                if st.button("SUPPRIMER",key=6):
                                    delete_data_Audit(name_delete)
                                    st.warning("SUPPRIMER: '{}'".format(name_delete))
                                
                                st.markdown('### DONNÉE MODIFIÉE')
                                df_Audit = pd.DataFrame(view_Audit(), columns=["id","IDD","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"])
                                IDD2 = email.strip('][').split(', ')

                                @st.cache
                                def Audit_2(df_Audit: pd.DataFrame) -> pd.DataFrame:
                                    df_Audit = df_Audit[(df_Audit["IDD"].isin(IDD2))]
                                    return df_Audit.loc[1:, ["id","Chantier","AC","VC","NEU","SMPAR","NPR","IE","Date"]]

                                df_Audit1 = Audit_2(df_Audit)
                                #filtrage par chantier
                                splitted_df_Audit1 = df_Audit1['Chantier'].str.split(',')
                                unique_vals8 = list(dict.fromkeys([y for x in splitted_df_Audit1  for y in x]).keys())
                                filtrechantier8 = st.selectbox('AFFICHEZ VOTRE GRILLE EN FONCTION DU CHANTIER', unique_vals8,key=36)
                                mask = (df_Audit1['Chantier'] == filtrechantier8)
                                df_filter8=df_Audit1.loc[mask]
                                st.dataframe(df_filter8)                                    
        

                            

                                                    

                                                       




                                
                























            else:
                st.warning("Veuillez-vous enregistrer")





    elif choice == "Inscription":
        st.subheader("Créer un nouveau compte")
        new_user = st.text_input("Email")
        new_password = st.text_input("Mot de passe",type='password')
        

        if st.button("Inscription"):
            #pour valider l'entrée email
            regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
            if(re.search(regex, new_user)):
                new_user
            else:
                st.error("Email non valide")
                st.stop()
            create_table()
            add_userdata(new_user,make_hashes(new_password))
            #initialisation de la base de donné pour l'application je l'ai incrusté ici rien avoir avec le code login
            IDD=new_user
            Chantier=0
            NArrivant=0
            Ninduction=0
            NChantier=0
            NTBM=0
            NCR=0
            FNCR=0
            NCC=0
            FNCC=0
            NCH=0
            FNCH=0
            NCHC=0
            FNCHC=0
            NA=0
            FNA=0
            NAC=0
            FNAC=0
            NAct=0
            NJSA=0
            NInc=0
            AAA=0
            ASA=0
            AT=0
            NJP=0
            AC=0
            VC=0
            NEU=0
            SMPAR=0
            NPR=0
            IE=0
            
            T1=(datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
            T2=(datetime.now() - timedelta(2)).strftime('%Y-%m-%d')
            Date=T2
            Date2=T1
            c.execute('INSERT INTO Accueil(IDD,Chantier,NArrivant,Ninduction,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NArrivant,Ninduction,Date2))
            conn.commit()
            c.execute('INSERT INTO Accueil(IDD,Chantier,NArrivant,Ninduction,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NArrivant,Ninduction,Date2))
            conn.commit()
            c.execute('INSERT INTO TBM(IDD,Chantier,NChantier,NTBM,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NChantier,NTBM,Date))
            conn.commit()
            c.execute('INSERT INTO TBM(IDD,Chantier,NChantier,NTBM,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NChantier,NTBM,Date2))
            conn.commit()
            c.execute('INSERT INTO NC(IDD,Chantier,NCR,FNCR,NCC,FNCC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NCR,FNCR,NCC,FNCC,Date))
            conn.commit()

            c.execute('INSERT INTO NC(IDD,Chantier,NCR,FNCR,NCC,FNCC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NCR,FNCR,NCC,FNCC,Date2))
            conn.commit()
            c.execute('INSERT INTO Changements(IDD,Chantier,NCH,FNCH,NCHC,FNCHC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NCH,FNCH,NCHC,FNCHC,Date))
            conn.commit()
            c.execute('INSERT INTO Changements(IDD,Chantier,NCH,FNCH,NCHC,FNCHC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NCH,FNCH,NCHC,FNCHC,Date2))
            conn.commit()
            c.execute('INSERT INTO Anomalies(IDD,Chantier,NA,FNA,NAC,FNAC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NA,FNA,NAC,FNAC,Date))
            conn.commit()
            c.execute('INSERT INTO Anomalies(IDD,Chantier,NA,FNA,NAC,FNAC,Date) VALUES (%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NA,FNA,NAC,FNAC,Date2))
            conn.commit()
            c.execute('INSERT INTO JSA(IDD,Chantier,NAct,NJSA,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NAct,NJSA,Date))
            conn.commit()
            c.execute('INSERT INTO JSA(IDD,Chantier,NAct,NJSA,Date) VALUES (%s,%s,%s,%s,%s)',(IDD,Chantier,NAct,NJSA,Date2))
            conn.commit()
            c.execute('INSERT INTO Incident_Accident(IDD,Chantier,NInc,AAA,ASA,AT,NJP,Date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NInc,AAA,ASA,AT,NJP,Date))
            conn.commit()
            c.execute('INSERT INTO Incident_Accident(IDD,Chantier,NInc,AAA,ASA,AT,NJP,Date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,NInc,AAA,ASA,AT,NJP,Date2))
            conn.commit()
            c.execute('INSERT INTO Audit(IDD,Chantier,AC,VC,NEU,SMPAR,NPR,IE,Date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,AC,VC,NEU,SMPAR,NPR,IE,Date))
            conn.commit()
            c.execute('INSERT INTO Audit(IDD,Chantier,AC,VC,NEU,SMPAR,NPR,IE,Date) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)',(IDD,Chantier,AC,VC,NEU,SMPAR,NPR,IE,Date2))
            conn.commit()
            ####fin
            st.success("Votre compte a été créé avec succès")
            st.info("Allez au menu de connexion pour vous connecter")
        col1, col2, col3 = st.beta_columns([1,6,1])
        with col2:
            st.image("http://cabinetnpm.com/wp-content/uploads/2020/02/t%C3%A9l%C3%A9chargement.png",width=200,)





image_ren ="""
<img src="https://1tpecash.fr/wp-content/uploads/elementor/thumbs/Renaud-Louis-osf6t5lcki4q31uzfafpi9yx3zp4rrq7je8tj6p938.png" alt="Avatar" style="vertical-align: middle;width: 100px;height: 100px;border-radius: 50%;" >
"""

st.sidebar.markdown(image_ren, unsafe_allow_html = True)
st.sidebar.markdown('**Auteur: Renaud Louis DAHOU**')
st.sidebar.markdown('Email:dahou.r@yahoo.com')
st.sidebar.markdown('[Linkedin](https://www.linkedin.com/in/dahou-renaud-louis-8958599a/)')
st.sidebar.warning('Pour tester HSE KPI RECORDER et faire des enregistrements, allez dans menu- connexion et mettez les informations de connexion ou inscrivez-vous si vous êtes nouveau.') #.\n Email:dahou.r@yahoo.com \n Mot de passe:lyne18
if __name__ == '__main__':
    main()
#suite
updater = Updater("1836903308:AAFE4kcYQ61hmpiGxJMeRP9B6WuG3DQj-Fk")
dispatcher = updater.dispatcher
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_command))
dispatcher.add_handler(MessageHandler(Filters.text, run_bot))

# Start the Bot
updater.start_polling()
updater.idle()
