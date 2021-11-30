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
from sklearn.svm import LinearSVC
from bot_config import BOT_CONFIG


# Получение данных для машинного обучения
X_text = []
y = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)


# Векторизация фраз
# Lemmitization

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3),stop_words = stopwords.words('french'))
X = vectorizer.fit_transform(X_text)

clf = LinearSVC()
clf.fit(X, y)



# Фильтрация исходной фразы
def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in 'dffgcgsxjhgsbxnsbcjdhgcdnshdckshvkjvhdskjhvjkdfvh']
    text = ''.join(text)
    return text.strip()


# Понять намерение, цель фразы
def get_intent(replica):
    replica = filter_text(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in BOT_CONFIG['intents'][intent]['examples']:
        example = filter_text(example)
        distance = edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


# Есть готовый ответ
def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses']
        #return responses
        if responses:
            return random.choice(responses)



# Болталка
# Парсим, фильтруем и оптимизируем датасет с диалогами
with open('dialogues.txt', encoding="utf-8") as f:
    content = f.read()

dialogues_str = content.split('\n\n')
dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]


dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue
    
    question, answer = dialogue
    question = filter_text(question[2:])
    answer = answer[2:]

    if question != '' and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])


dialogues_structured = {}

for question, answer in dialogues_filtered:
    words = set(question.split(' '))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])


dialogues_structured_cut = {}

for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]



# Придумать свой ответ
def generate_answer_by_text(replica):
    replica = filter_text(replica)
    words = set(replica.split(' '))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
             mini_dataset += dialogues_structured_cut[word]
    
    answers = [] # [[proba, question, answer], [], ...]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            dist = edit_distance(replica, question)
            dist_weighted = dist / len(question)
            if dist_weighted < 0.2:
                answers.append([dist_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]



# Фразы заглушки, когда нет готового ответа и нельзя придумать свой ответ
def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


stats = {'intent': 0, 'generate': 0, 'failure': 0}


def bot(replica):
    # NLU
    intent = get_intent(replica)

    # Get answer

    # Ready answer
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer

    # Generate answer
    answer = generate_answer_by_text(replica)
    if answer:
        stats['generate'] += 1
        return answer

    # Use stub
    stats['failure'] += 1
    return get_failure_phrase()




from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext


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
    replica = update.message.text
    answer = bot(replica)
    update.message.reply_text(answer)


def main() -> None:
    """Start the bot."""
    updater = Updater("1836903308:AAFE4kcYQ61hmpiGxJMeRP9B6WuG3DQj-Fk")

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, run_bot))

    # Start the Bot
    updater.start_polling()
    updater.idle()

main()
