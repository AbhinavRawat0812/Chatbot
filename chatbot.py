import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import sys
import cv2
from PIL import Image

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pk1','rb'))
classes = pickle.load(open('classes.pk1','rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i]=1

    return np.array(bag)

def predict_class(sentence):
    bow=bag_of_words(sentence)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def get_response(intents_list,intents_json):
    result=[]
    should_display_img = "yes"
    tag = intents_list[0]['intent']
    probab = intents_list[0]['probability']
    probab = float(probab)
    # print(probab)
    if probab < 0.98:
        result.append("Sorry I didn't got it! Please try again.")
        result.append("false")
        result.append("xyz")
        return result
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result.append(random.choice(i['responses']))
            if i['displayimg'][0] == should_display_img:
                result.append(i['displayimg'][1])
            else:
                result.append("false")
            result.append(tag)
            break
    return result

print("\n\t\t\t\t\tThis is CarEx your car expert guide! \n")
while True:
    message = input("\nYOU: ")
    ints = predict_class(message)
    res=get_response(ints, intents)
    print("BOT: "+res[0]+"\n")
    if res[1]!= "false":
        img = Image.open(res[1])
        want_to_display = input("\n\n*****  Do you wish to see the visuals - YES/NO  *****\nYOU: ")
        if want_to_display == "yes" or want_to_display == "YES":
            img.show()
    if res[2] == "goodbye":
        break
        