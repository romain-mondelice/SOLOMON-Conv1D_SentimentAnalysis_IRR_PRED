# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:29:17 2021
@author: Romain
"""

"""
Implementation of Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin & Vedant Misra paper on finance data
GROKKING: GENERALIZATION BEYOND OVERFITTING ON SMALL ALGORITHMIC DATASETS
"""

import pandas as pd
import numpy as np

import pytz
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import MetaTrader5 as mt5

# Import Libraries
from textblob import TextBlob
import sys
import tweepy
import os
import nltk
import pycountry
import re
import string
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Authentication
consumerKey = "v0uet06HY4pxro9DJ6e92qMRC"
consumerSecret = "DrKUX5f4nH0NSpazSCwJxxePotjn3LpUfPbrOmvoyKSoVm3wV2"
accessToken = "1360044026066526213-xhug05dsJjWU6MftrKkia2qCzG9Qsd"
accessTokenSecret = "0iAmMBdRwsxpN4aGnlqj6HLym0UTrVeQBFXtRd1uqDok3"
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

"""
##############################################################################
# HyperParameters => TODO (find the best combinaison of hyperparameters)
##############################################################################
"""

#Use to set the data lenhgt
year_lenght = 20

lookback_window_size = 3

#TimeFrame that we use for our FOREX data
timeframe = mt5.TIMEFRAME_W1

#Week size associated to the timeframe ex: H4 => 5 days => 1 week => 30 H4
size_week = 1 #Avec 0.1

#Nb week on what we want to train
nb_week = 500

#Symbol => If other that EURUSD should change commission rate for more precision
symbol = "AMZN" #have to be the same name that in AdmiralMarket

dates = [[2021, 1, 2]]

all_closes = []
all_low = []
all_high = []
preds_qtstats = []
preds_validation = []
preds_list = []
positive_list_final = []
negative_list_final = []
neutral_list_final = []

for date in dates:
    
    """
    ##############################################################################
    # Init of Solomon
    ##############################################################################
    """
    all_preds = []
    
    #Here the year / month / day / hour / minutes that we are going to
    #See utc_from / utc_to
    year = date[0]
    month = date[1]
    day = date[2]
    hour = 0
    minutes = 0
    
    print("\n=============================================================================")
    print(f"Inference Solomon - {day}-{month}-{year} // {nb_week}")
    print("=============================================================================")
    
    # establish connection to MetaTrader 5 terminal
    if not mt5.initialize():
        print("initialize() failed, error code =",mt5.last_error())
        quit()
    
    # set time zone to UTC
    timezone = pytz.timezone("Etc/UTC")
    
    # create 'datetime' objects in UTC time zone to avoid the implementation of a local time zone offset
    import datetime
    utc_from = datetime.datetime(year-year_lenght, month, day, tzinfo=timezone)
    utc_to = datetime.datetime(year, month, day, hour, minutes, tzinfo=timezone)
    
    #Get user input
    query = "amazon"
    noOfTweet = 500
    noOfDays = 5
    
    #Creating list to append tweet data
    tweets_list = []
    
    now = utc_to.strftime('%Y-%m-%d')
    yesterday = utc_to - datetime.timedelta(days = int(noOfDays))
    yesterday = yesterday.strftime('%Y-%m-%d')
    import snscrape.modules.twitter as sntwitter
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(query + ' lang:en since:' +  yesterday + ' until:' + now + ' -filter:links -filter:replies').get_items()):
        if i > int(noOfTweet):
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username])
    
    #Creating a dataframe from the tweets list above
    df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    
    # Create a function to clean the tweets
    def cleanTxt(text):
        text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
        text = re.sub('#', '', text) # Removing '#' hash tag
        text = re.sub('RT[\s]+', '', text) # Removing RT
        text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
        return text
    
    #applying this function to Text column of our dataframe
    df["Text"] = df["Text"].apply(cleanTxt)
    
    df = df[df["Text"].str.contains("Closed")==False]
    df = df[df["Text"].str.contains("opened")==False]
    df = df[df["Text"].str.contains("closed")==False]
    df = df[df["Text"].str.contains("Bought")==False]
    df = df[df["Text"].str.contains("Sold")==False]
    
    #Sentiment Analysis
    def percentage(part,whole):
        return 100 * float(part)/float(whole)
    
    #Assigning Initial Values
    positive = 0
    negative = 0
    neutral = 0
    
    #Creating empty lists
    tweet_list1 = []
    neutral_list = []
    negative_list = []
    positive_list = []
 
    #Iterating over the tweets in the dataframe
    from textblob import TextBlob
    from textblob_fr import PatternTagger, PatternAnalyzer
    for tweet in df['Text']:
        tweet_list1.append(tweet)
        blob = TextBlob(tweet)
        sentiment = blob.sentiment[0]
    
        if sentiment < 0:
            negative_list.append(tweet) #appending the tweet that satisfies this condition
            negative += 1 #increasing the count by 1
        elif sentiment > 0:
            positive_list.append(tweet) #appending the tweet that satisfies this condition
            positive += 1 #increasing the count by 1
        elif sentiment == 0:
            neutral_list.append(tweet) #appending the tweet that satisfies this condition
            neutral += 1 #increasing the count by 1 
    
    positive = percentage(positive, len(df)) #percentage is the function defined above
    negative = percentage(negative, len(df))
    neutral = percentage(neutral, len(df))
    
    #Converting lists to pandas dataframe
    tweet_list1 = pd.DataFrame(tweet_list1)
    neutral_list = pd.DataFrame(neutral_list)
    negative_list = pd.DataFrame(negative_list)
    positive_list = pd.DataFrame(positive_list)
    
    #using len(length) function for counting
    print("Since " , noOfDays , " days, there have been", len(tweet_list1) ,  "tweets on " + query, end='\n*')
    print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n*')
    print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n*')
    print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n*')
    
    positive_list_final.append(len(positive_list))
    neutral_list_final.append(len(neutral_list))
    negative_list_final.append(len(negative_list))
    
    labels = ['Positive ['+str(round(positive))+'%]' , 'Neutral ['+str(round(neutral))+'%]','Negative ['+str(round(negative))+'%]']
    sizes = [positive, neutral, negative]
    colors = ['yellowgreen', 'blue','red']
    patches, texts = plt.pie(sizes,colors=colors, startangle=90)
    plt.style.use('default')
    plt.legend(labels)
    plt.title("Sentiment Analysis Result for keyword= "+query+"" )
    plt.axis('equal')
    plt.show()
    
    # word cloud visualization
    def word_cloud(text):
        stopwords = set(STOPWORDS)
        allWords = ' '.join([twts for twts in text])
        wordCloud = WordCloud(background_color='black',width = 1600, height = 800,stopwords = stopwords,min_font_size = 20,max_font_size=150,colormap='prism').generate(allWords)
        fig, ax = plt.subplots(figsize=(20,10), facecolor='k')
        plt.imshow(wordCloud)
        ax.axis("off")
        fig.tight_layout(pad=0)
        plt.show()
    
    print('Wordcloud for ' + query)
    word_cloud(df['Text'].values)
    
    #Vectorization for Data Visualization
    def vectorization(table):
        #CountVectorizer will convert a collection of text documents to a matrix of token counts
        #Produces a sparse representation of the counts 
        #Initialize
        vector = CountVectorizer()
        #We fit and transform the vector created
        frequency_matrix = vector.fit_transform(table.tweet)
        #Sum all the frequencies for each word
        sum_frequencies = np.sum(frequency_matrix, axis=0)
        #Now we use squeeze to remove single-dimensional entries from the shape of an array that we got from applying np.asarray to
        #the sum of frequencies.
        frequency = np.squeeze(np.asarray(sum_frequencies))
        #Now we get into a dataframe all the frequencies and the words that they correspond to
        frequency_df = pd.DataFrame([frequency], columns=vector.get_feature_names()).transpose()
        return frequency_df
    
    def graph(word_frequency, sent):
        labels = word_frequency[0][1:51].index
        title = "Word Frequency for %s" %sent
        #Plot the figures
        plt.figure(figsize=(10,5))
        plt.bar(np.arange(50), word_frequency[0][1:51], width = 0.8, 
                color = sns.color_palette("bwr"), alpha=0.5, 
                edgecolor = "black", capsize=8, linewidth=1);
        plt.xticks(np.arange(50), labels, rotation=90, size=14);
        plt.xlabel("50 more frequent words", size=14);
        plt.ylabel("Frequency", size=14);
        #plt.title('Word Frequency for %s', size=18) %sent;
        plt.title(title, size=18)
        plt.grid(False);
        plt.gca().spines["top"].set_visible(False);
        plt.gca().spines["right"].set_visible(False);
        plt.show() 
    
    
    # EURUSD Dataframe
    rates = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
    
    rates_frame = pd.DataFrame(rates)
    
    rates_frame['time']=pd.to_datetime(rates_frame['time'], unit='s')
    
    FOREX_df = rates_frame
    
    FOREX_df = FOREX_df.drop(columns=['spread', 'real_volume'])
    
    """
    # =============================================================================
    # Define the data on wich we have trained in order to fit_transform on them
    # =============================================================================
    """
    
    df = FOREX_df.copy()
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = df['Date'].values.astype(np.datetime64)
    df['Volume'] = df['Volume'].values.astype(int)
    
    # =============================================================================
    # Calculate the mean TP available
    # =============================================================================
    mean_high = (df["High"] - df["Close"]).mean()
    mean_low = (df["Close"] - df["Low"]).mean()
    
    # =============================================================================
    # init train_df
    # =============================================================================
    total_df_size = size_week*nb_week
    train_df = df[len(df)-total_df_size:].copy()
    
    # =============================================================================
    # Classify with next close    
    # =============================================================================
    
    #Get the min and max of the week following the current close price
    train_df["Min"] = train_df['Low'].shift(-1).rolling(window=1).min()
    train_df["Max"] = train_df['High'].shift(-1).rolling(window=1).max()
    
    #Achieve a percentage change in price
    #Example for the max: If the percentage change is high it means that the high point of 
    # the following week is far from the current price close
    train_df["Pct Change Min"] = ((train_df["Min"] - train_df['Close']) / train_df['Close']) * 100
    train_df["Pct Change Max"] = ((train_df["Max"] - train_df['Close']) / train_df['Close']) * 100
    
    #Then we convert our price changes into percentages to get a better overview
    train_df["Up Ratio"] = (abs(train_df["Pct Change Max"]) / (abs(train_df["Pct Change Min"]) + abs(train_df["Pct Change Max"]))) * 100
    train_df["Down Ratio"] = (abs(train_df["Pct Change Min"]) / (abs(train_df["Pct Change Min"]) + abs(train_df["Pct Change Max"]))) * 100
    
    #Finally we choose a ratio that we will classify
    #Attention according to the chosen ratio (Up or Down) the actions are reversed
    train_df['INFERENCE-RiskRatio'] = train_df["Up Ratio"]
    
    #We drop the useless columns so as not to disturb the learning process with data that already come from the future
    train_df = train_df.drop(columns=['Date', 'Min', 'Max', 'Pct Change Min', 'Pct Change Max', 'Up Ratio', 'Down Ratio'])
    
    # =============================================================================
    # Drop nans    
    # =============================================================================
    
    train_df = train_df.dropna()
    
    train_df["INFERENCE-RiskRatio"] = train_df["INFERENCE-RiskRatio"].astype(int)
    
    def classify(irr):
        if (irr > 50):
            return 1
        elif (irr < 50):
            return 0
        else:
            return 2
        
    train_df['INFERENCE-RiskRatio'] = list(map(classify, train_df['INFERENCE-RiskRatio']))
    
    # =============================================================================
    # Scale train_df    
    # =============================================================================
    from sklearn.preprocessing import MinMaxScaler
    
    price_sc = MinMaxScaler(feature_range=(0, 10))
    volume_sc = MinMaxScaler(feature_range=(0, 10))
    ir_sc = MinMaxScaler(feature_range=(0, 10))
    
    train_df['Open'] = price_sc.fit_transform(train_df["Open"].values.reshape(-1,1))
    train_df['High'] = price_sc.transform(train_df["High"].values.reshape(-1,1))
    train_df['Low'] = price_sc.transform(train_df["Low"].values.reshape(-1,1))
    train_df['Close'] = price_sc.transform(train_df["Close"].values.reshape(-1,1))
    train_df['Volume'] = volume_sc.fit_transform(train_df["Volume"].values.reshape(-1,1))
    #train_df['INFERENCE-RiskRatio'] = ir_sc.fit_transform(train_df["INFERENCE-RiskRatio"].values.reshape(-1,1))
    
    train_df = train_df.dropna()
    train_df = train_df.reset_index(drop=True)
    
    all_closes.append(train_df["Close"].iloc[-1:].values)
    all_low.append(train_df["Low"].iloc[-1:].values)
    all_high.append(train_df["High"].iloc[-1:].values)
    
    #We drop the useless columns so as not to disturb the learning process with data that already come from the future
    #train_df = train_df.drop(columns=['Open', 'High', 'Low', 'Volume'])
    
    # =============================================================================
    # Dropna
    # =============================================================================
    
    train_df = train_df.astype(int)
    
    # =============================================================================
    # Create custom columns
    # =============================================================================
    train_df['PrevINFERENCE-RiskRatio'] = train_df['INFERENCE-RiskRatio'].shift(1)
    train_df = train_df.dropna()
    
    # =============================================================================
    # Reorder column
    # =============================================================================
    column_names = ["Open", "High", "Low", "Close", "Volume", "PrevINFERENCE-RiskRatio", "INFERENCE-RiskRatio"]
    train_df = train_df.reindex(columns=column_names)
    
    # =============================================================================
    # Save train_df 
    # =============================================================================
    
    train_df.to_csv('C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/train_df.csv', index = False)
    
    """
    ######################################################################################
    # Build the binary operation tables
    ######################################################################################
    """
    
    binary_tables = pd.DataFrame(index=pd.unique(train_df["Close"]), columns=pd.unique(train_df['PrevINFERENCE-RiskRatio']).astype(int))
    
    for irr in train_df['INFERENCE-RiskRatio']:
        res = train_df.loc[train_df['INFERENCE-RiskRatio'] == irr]
        for row in res.iterrows():    
            binary_tables.at[row[1]["Close"], row[1]["PrevINFERENCE-RiskRatio"]] = irr
    
    binary_tables = binary_tables.drop(binary_tables.columns[0], axis=1)
    binary_tables = binary_tables.drop(binary_tables.columns[-1], axis=1)
    
    # =============================================================================
    # Finish the preparation of train data
    # =============================================================================
    train_array = np.array(train_df)
    final_train_array = train_array
    
    train_total_x = []
    train_total_y = []
    
    for i in range(lookback_window_size, len(final_train_array)):
        train_total_x.append(final_train_array[(i-(lookback_window_size))+1:i+1, :train_df.shape[1]-1])
        train_total_y.append(final_train_array[i, -1])
    
    (unique, counts) = np.unique(train_total_y, return_counts=True)
    preds_df = pd.DataFrame(index=range(len(train_total_y)),columns=range(len(unique)))
    preds_df = preds_df.fillna(0)
    
    for i in range(len(preds_df)):
        pred = int(train_total_y[i])
        preds_df[pred].iloc[i] = 1
    
    total_train = list(zip(train_total_x, train_total_y))
    
    #Random to prevent the strong influence of previous weeks
    #import random
    #random.shuffle(total_train)
    
    train_x, train_y = zip(*total_train)
    
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    train_y_concatenated = np.array(preds_df)
    
    validation_x = train_x[len(train_x)-1:]
    validation_y = train_y[len(train_y)-1:]
    validation_y_concatenated = train_y_concatenated[len(train_y_concatenated)-1:]
    
    train_x = train_x[:len(train_x)-1]
    train_y = train_y[:len(train_y)-1]
    train_y_concatenated = train_y_concatenated[:len(train_y_concatenated)-1]
    
    """
    ######################################################################################
    # Build model
    ######################################################################################
    """
    from tensorflow import keras
    import tensorflow_addons as tfa
    def build_model(input_shape, nb_classes):
        n_feature_maps = 64
    
        input_layer = keras.layers.Input(input_shape)
    
        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
    
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
    
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
    
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)
    
        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
    
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
    
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
    
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    
        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)
    
        # BLOCK 3
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
    
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
    
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
    
        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)
    
        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
    
        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
    
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
        base_learning_rate = 1e-5
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(base_learning_rate),
                      metrics=['accuracy'])
    
        return model
    
    """
    ######################################################################################
    # Train model
    ######################################################################################
    """
    #model = build_model(train_x.shape[1:], train_y_concatenated.shape[1])
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(train_x.shape[1:]),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.25),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.125),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(3, activation='softmax'),
    ])
    
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="auto", name="sparse_categorical_crossentropy"),
              optimizer=tf.keras.optimizers.Adam(1e-3),
              metrics=['sparse_categorical_accuracy'])
    
    history = model.fit(train_x, 
                        train_y,
                        batch_size = 64,
                        epochs=200,
                        validation_data=(validation_x, validation_y))
    
    preds_ = model.predict(validation_x)
    preds_list.append(preds_)
    preds_validation.append(validation_y)

# =============================================================================
# Create df from data to test strategy and get metrics
# =============================================================================
preds_qtstats = []
for pred in preds_list:
    print(pred)
    if(pred[0][0] > 0.50):
        preds_qtstats.append([0])
    elif(pred[0][1] > 0.50):
        preds_qtstats.append([1])
    else:
        preds_qtstats.append([2])
        
all_closes_merged = [item for sublist in all_closes for item in sublist]
all_high_merged = [item for sublist in all_high for item in sublist]
all_low_merged = [item for sublist in all_low for item in sublist]

all_closes_merged = price_sc.inverse_transform(np.array(all_closes_merged).reshape(-1,1)).tolist()
all_closes_merged = [item for sublist in all_closes_merged for item in sublist]
all_high_merged = price_sc.inverse_transform(np.array(all_high_merged).reshape(-1,1)).tolist()
all_high_merged = [item for sublist in all_high_merged for item in sublist]
all_low_merged = price_sc.inverse_transform(np.array(all_low_merged).reshape(-1,1)).tolist()
all_low_merged = [item for sublist in all_low_merged for item in sublist]

preds_qtstats_merged = [item for sublist in preds_qtstats for item in sublist]
preds_validation_merged = [item for sublist in preds_validation for item in sublist]

df_closes = pd.DataFrame(all_closes_merged, columns=['Close'])
df_closes["High"] = all_high_merged
df_closes["Low"] = all_low_merged
df_closes['Predicted INFERENCE-RiskRatio'] = preds_qtstats_merged
df_closes['Predicted INFERENCE-RiskRatio'] = df_closes['Predicted INFERENCE-RiskRatio'].astype(int)
df_closes['Positive'] = positive_list_final
df_closes['Negative'] = negative_list_final
df_closes['Neutral'] = neutral_list_final

# =============================================================================
# Get the real INFERENCE RiskRatio
# =============================================================================
#Finally we choose a ratio that we will classify
#Attention according to the chosen ratio (Up or Down) the actions are reversed
df_closes['Real INFERENCE-RiskRatio'] = preds_validation_merged

# =============================================================================
# Calculate when INFERENCE-RiskRatio is in the good trend
# =============================================================================

def check(predicted_inference_riskratio, real_inference_riskratio):
    if(predicted_inference_riskratio == real_inference_riskratio):
        return 1
    else:
        return 0
    
df_closes["Check"] = list(map(check, df_closes["Predicted INFERENCE-RiskRatio"], df_closes["Real INFERENCE-RiskRatio"]))

preds_ok = sum(df_closes["Check"].dropna())

preds_ok_percentage = (preds_ok / (len(df_closes["Check"].dropna())))*100

# =============================================================================
# Save df_closes 
# =============================================================================

df_closes.to_excel('C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/preds_df.xlsx')

# =============================================================================
# Calculate cumulative returns
# =============================================================================
plt.clf()
plt.figure(figsize=(20, 10))
plt.plot(df_closes['Close'], "-p", label="current_close")

current_pos = None
short_close = None
long_close = None
count_days = 0
cumulutative_return = []
for i in range(len(df_closes)):
    if(df_closes["Predicted INFERENCE-RiskRatio"].iloc[i] == 1):
        if(current_pos == 1):
            #plt.scatter(i, df_closes['Close'].iloc[i], c='purple', label='buy', s = 80, edgecolors='none', marker="^")
            pass
        else:
            current_pos = 1
            plt.scatter(i, df_closes['Close'].iloc[i], c='green', label='buy', s = 120, edgecolors='none', marker="^")
            long_close = df_closes["Close"].iloc[i]
            #Calculate return
            if(short_close != None):
                ret = short_close - long_close
                cumulutative_return.append(ret)
    elif(df_closes["Predicted INFERENCE-RiskRatio"].iloc[i] == 0):
        if(current_pos == 0):    
            #plt.scatter(i, df_closes['Close'].iloc[i], c='purple', label='sell', s = 80, edgecolors='none', marker="v")
            pass
        else:
            current_pos = 0
            plt.scatter(i, df_closes['Close'].iloc[i], c='red', label='sell', s = 120, edgecolors='none', marker="v")
            short_close = df_closes["Close"].iloc[i]
            if(long_close != None):
                ret = short_close - long_close
                cumulutative_return.append(ret)

import statistics
volatility = int(statistics.stdev(all_closes_merged) * 100000)
total_cumulutative_return = int(sum(cumulutative_return) * 100000)
sharpe = total_cumulutative_return / volatility

print("--------------------------------------------------------------------------")
print(f"- Cumulative Returns : {total_cumulutative_return} points")
print(f"- Mean Cumulative Returns by week : {int(total_cumulutative_return/len(dates))} points")
print(f"- Sharpe Ratio : {round(sharpe, 5)}")
print(f"- Preds percentage OK : {round(preds_ok_percentage, 2)}")
print("--------------------------------------------------------------------------\n")

plt.ylabel('Prices')
plt.xlabel('Timesteps')
plt.title(f'Final test graph {round(sharpe, 5)} - {round(total_cumulutative_return, 5)}')
plt.savefig(f"C:/Users/romai/Desktop/INFERENCE/PRODUITS/INFERENCE - SOLOMON/solomon/Plot reward mean over episode/final-test_plot_{day}-{month}-{year}.png")



