import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from collections import Counter
from textblob import TextBlob
import re


def labelling(file_name):
    data= pd.read_csv(file_name)

    for i in range(data.shape[0]):
    # remove special characters
        data.iloc[i]['blog_text'] = re.sub(r'\W', ' ', str(data.iloc[i]['blog_text']))
    # delete all single characters
        data.iloc[i]['blog_text']= re.sub(r'\s+[a-zA-Z]\s+', ' ', data.iloc[i]['blog_text'])
    # remove extra space
        data.iloc[i]['blog_text'] = re.sub(r'\s+', ' ', data.iloc[i]['blog_text'], flags=re.I)
    # uppercase to lowercase
        data.iloc[i]['blog_text'] = data.iloc[i]['blog_text'].lower()


####################Function for unsupervised sentiment analysis####################
    def get_textBlob_score(sent):
        polarity = TextBlob(sent).sentiment.polarity
        return polarity
###########Evaluating the sentiment for each tweet###############
    sentiment = []
    for i in range(data.shape[0]):
        h = data.iloc[i]
        comment = h['blog_text']
        sent_score = get_textBlob_score(comment)
        if sent_score> 0 :
            sentiment.append(1)
            try:
                positive_file = open("positive.txt","a")
                positive_file.writelines(comment)
                positive_file.close()
            except:
                continue
            
        elif sent_score == 0 :
            sentiment.append(0)
            try:
                neutral_file = open("neutral.txt","a")
                neutral_file.writelines(comment)
                neutral_file.close()
            except:
                continue
            
        else:
            sentiment.append(2)
            try:
                negative_file = open("negative.txt","a")
                negative_file.writelines(comment)
                negative_file.close()
            except:
                continue
            
    data['sentiment'] = sentiment
    data = data.drop(["url","user","date","time"], axis=1)
    return data

air_canada = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_aircanada.csv")
bell = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_bell.csv")
crave_canada = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_cravecanada.csv")
netflix = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_netflix.csv")
rogers = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_rogers.csv")
telus = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_telus.csv")
westjet = labelling("D:/Yokhesh/semester2/data_analytics/project/data/twitter_westjet.csv")

result = pd.concat([air_canada,bell,crave_canada,netflix,rogers,telus,westjet])
result = result.drop(["account"], axis=1)

