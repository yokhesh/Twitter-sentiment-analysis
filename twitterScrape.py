# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 11:24:56 2019

@author: User
"""
import platform
import os
import pandas as pd
pd.set_option('precision', 6)
from PySide2 import QtCore
import pickle
import numpy as np
import pandas as pd
import time
import nltk
import sqlite3

from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from collections import Counter
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,NavigationToolbar2QT as NavigationToolbar) # N
from wordcloud import WordCloud, STOPWORDS
from nltk.tokenize import word_tokenize

from helpers import scroll_tweets, twitter_page_extract, company_accounts

class twitterScrape(QtCore.QThread):
	
	progressEvent = QtCore.Signal(str)
	plotEvent = QtCore.Signal(object, int)
	
	def __init__(self):	
		QtCore.QThread.__init__(self)
		
		self.company_name = []
		self.output_path = []
		self.application_path = []
		self.bsLimit = []
		self.database_path = []
		self.graphicsView = []
		
		self.running = False
		
	def stop(self):
		self.running = False
	
	def create_word_cloud(self, string, typ, idx):
		stop_words = set(STOPWORDS)
		stop_words.add("netflix")
		stop_words.add("twitter")
		cloud = WordCloud(width=500, height=500,background_color = "white", max_words = 200, stopwords = stop_words)
		cloud.generate(string)
		self.fig = Figure()
		self.axes = self.fig.add_subplot()
#		plt.figure(figsize=(4,3))
		self.axes.imshow(cloud)
		self.axes.set_title(typ)
#	   plt.show()

		#### Add Plot To Graphics Area
		self.canvas = FigureCanvas(self.fig)
		
		self.plotEvent.emit(self.canvas,idx)
		
#		graphicscene = QtWidgets.QGraphicsScene()
#		graphicscene.addWidget(self.canvas)
#		
#		self.graphicsView.setScene(graphicscene)
#		self.graphicsView.show()
		
	def clou(self, dataset, typ, idx):
		typ = typ
		try:
			words = set(nltk.corpus.words.words())
		except:
			nltk.download('all-nltk')
			words = set(nltk.corpus.words.words())
			
		################removing unrecognised words
		dataset_clean =  " ".join(w for w in nltk.wordpunct_tokenize(dataset) \
				 if w.lower() in words or not w.isalpha())
		################tokenzing into separate words
		try:
			tokens=word_tokenize(dataset_clean)
		except:
			nltk.download('words')
			nltk.download('all-nltk')
			tokens=word_tokenize(dataset_clean)
		
		################tags for each word
		pos = nltk.pos_tag(tokens)
		allowed_type = ['JJ','JJR','JJS','RB','RBR','RBS']
		req_word = []
		##############storing only the words that match the required type
		for i in range(len(pos)):
				if pos[i][1] in allowed_type:
						req_word.append(pos[i][0])
		#############converting list to string
		req_str = ' '.join(req_word)
		
		self.create_word_cloud(req_str,typ, idx)
	
	def add_value_labels(self,ax, spacing=5):
		for rect in ax.patches:
			# Get X and Y placement of label from rect.
			y_value = rect.get_height()
			x_value = rect.get_x() + rect.get_width() / 2
			space = spacing
			va = 'bottom'
	
			# Use Y value as label and format number with one decimal place
			label = "{:d}".format(y_value)
	
			# Create annotation
			ax.annotate(
				label,                      # Use `label` as label
				(x_value, y_value),         # Place label at end of the bar
				xytext=(0, space),          # Vertically shift label by `space`
				textcoords="offset points", # Interpret `xytext` as offset in points
				ha='center',                # Horizontally center label
				va=va)                      # Vertically align label differently for
											# positive and negative values.
										
	def predictiveModel(self, data):
		text = data['blog_text']
		
		X_test1 = pd.DataFrame(text)
		with open(os.path.join(self.application_path, 'tokenizer.pickle'), 'rb') as handle:
			tokenizer = pickle.load(handle)
		
		X_test = tokenizer.texts_to_sequences(text)
		
		maxlen = 100
		X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
		
		######Loading the saved_model##########
		json_file = open(os.path.join(self.application_path, 'model.json'), 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights(os.path.join(self.application_path,'model.h5'))
		loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		#loss, accuracy = loaded_model.evaluate(X_test, y_test, verbose=False)
		pred = loaded_model.predict(X_test)
		
		#########Word cloud######
		self.statusUpdateText = 'Starting word cloud generation ...'
		self.progressEvent.emit(self.statusUpdateText)
		try:
			words = set(nltk.corpus.words.words())
		except:
			self.statusUpdateText = 'Performing one-time install of nltk words ...'
			self.progressEvent.emit(self.statusUpdateText)
			nltk.download('all-nltk')
			words = set(nltk.corpus.words.words())
			self.statusUpdateText = 'Done installing nltk words.'
			self.progressEvent.emit(self.statusUpdateText)
				
		pred_out = list(np.argmax(pred,axis = 1))
		sent_val = []
		for i in range(pred.shape[0]):
			sent_score = pred_out[i]
			comment = X_test1.iloc[i]['blog_text']
			
			if sent_score == 0 :
				sent_val.append('Neutral')
				try:
					positive_file = open(os.path.join(self.output_path,"neutral.txt"),"a")
					positive_file.writelines(comment)
					positive_file.close()
				except:
					continue
			elif sent_score == 1 :
				sent_val.append('Positive')
				try:
					neutral_file = open(os.path.join(self.output_path, "positive.txt"),"a")
					neutral_file.writelines(comment)
					neutral_file.close()
				except:
					continue
			else:
				sent_val.append('Negative')
				try:
					negative_file = open(os.path.join(self.output_path,"negative.txt"),"a")
					negative_file.writelines(comment)
					negative_file.close()
				except:
					continue
		
		x_pos= list(Counter(sent_val).keys())
		x_pos1 = np.arange(len(x_pos))
		y_pos = list(Counter(sent_val).values())
		
		###############Plotting the number of positive,negative,neutral tweets###########
		border = 25
		dpi = 80
		figsize= ((700-100)+2*border)/float(dpi), ((400-100)+2*border)/float(dpi)
		self.fig = Figure(figsize=figsize, dpi=dpi)
		self.axes = self.fig.add_subplot()

		self.axes.bar(x_pos1, y_pos, align='center', alpha=0.5)
		self.axes.set_xticks(x_pos1)
		self.axes.set_xticklabels(x_pos)
		self.axes.set_ylabel('Number of Tweets')
		self.add_value_labels(self.axes)
		ylimits = self.axes.get_ylim()
		self.axes.set_ylim([ylimits[0],ylimits[1]+50])
		
		#### Add Plot To Graphics Area
		self.canvas = FigureCanvas(self.fig)
		
		self.plotEvent.emit(self.canvas, 1)
		
		##############Creatng the wordcloud###########
		dataset_pos = open(os.path.join(self.output_path,"positive.txt"), "r").read()
		dataset_neg = open(os.path.join(self.output_path,"negative.txt"), "r").read()
		dataset_neu = open(os.path.join(self.output_path,"neutral.txt"), "r").read()
		
		self.clou(dataset_pos,'Positive Tweets', 2)
		self.clou(dataset_neg,'Negative Tweets', 3)
		
	
	def createDatabase(self):
		# Create table in db
		connection = sqlite3.connect(self.database_path)
		cursor = connection.cursor()
		
		sql_command = """CREATE TABLE companies (id INT PRIMARY KEY,
							company TEXT, 
							handles TEXT);"""
		cursor.execute(sql_command)
		
		sql_command = """CREATE TABLE tweet_data (id INT, account TEXT, url TEXT, 
							user TEXT, date TEXT, time TEXT, blog_text TEXT);"""
		cursor.execute(sql_command)
		
		connection.commit()
		connection.close()
	
	def databaseCheck(self):
		connection = sqlite3.connect(self.database_path)
		cursor = connection.cursor()
		
		cursor.execute("SELECT company FROM companies") 
		result = cursor.fetchall()
		if not any(x[0].lower() == self.company_name.lower() for x in result):
			return False
		else:
			return True
		
	def queryDatabase(self, data):
		connection = sqlite3.connect(self.database_path)
		cursor = connection.cursor()
		if len(data)>0:
			cursor.execute("SELECT company FROM companies") 
			result = cursor.fetchall()
			id_number = len(result)+1
			if 'account' not in list(data):
				data['account'] = np.repeat(self.company_name, len(data))
				twitter_handle = np.unique(data['account'])
			else:
				twitter_handle = np.unique(data['account'])
				
			sql_data = [id_number, self.company_name, twitter_handle]
			data['id'] = np.repeat(id_number, len(data))
		
			connection = sqlite3.connect(self.database_path)
			cursor = connection.cursor()
		
			format_str = """INSERT INTO companies (id, company, handles)
							VALUES ("{id_number}","{company_name}", "{twitter_handles}");"""
			sql_command = format_str.format(id_number = sql_data[0], company_name=sql_data[1], twitter_handles=sql_data[2])
			cursor.execute(sql_command)
		
			# Insert whole DataFrame into MySQL
			data.to_sql('tweet_data', con = connection, if_exists = 'append', chunksize = 1000, index=False)
		else:
			company = pd.read_sql_query("select * from companies;", connection)
			company_idx = company[company.company.str.lower() == self.company_name.lower()]['id'].values
			data = pd.read_sql_query("select * from tweet_data;", connection)
			data = data[data['id'] == int(company_idx)]
			
		connection.commit()
		connection.close()
		
		return data
	
	path_to_chromedriver = os.path.join(r'D:\PhD\2sem\sentiment_analysis_paper\ECE9603_project-master\gui\chrome_driver', 'chromedriver.exe')
	company_name = 'Rogers'
	@QtCore.Slot()
	def run(self):
		
		self.running = True
		
		# Download the chromedriver from (https://sites.google.com/a/chromium.org/chromedriver/home)
		# and place it somewhere on your PATH. Indicate location here:
		if platform.system() == 'Windows':
			self.chrome_exe = 'chromedriver.exe'
		elif platform.system().lower() == 'linux':
			self.chrome_exe = 'chromedriver_linux'
			os.chmod(os.path.join(self.application_path, 'chrome_driver', self.chrome_exe), 755)
		elif platform.system() == 'Darwin':
			self.chrome_exe = 'chromedriver_mac'
			
		self.path_to_chromedriver = os.path.join(self.application_path, 'chrome_driver', self.chrome_exe)
		self.database_path = os.path.join(self.application_path, 'twitter_data.db')
		if not os.path.exists(self.database_path):
			self.createDatabase()
			
		while self.running:
			in_database = self.databaseCheck()
			if not in_database:
# 				other_accounts = company_accounts(self.company_name, self.path_to_chromedriver)
# 				all_accounts = [self.company_name] + other_accounts
				other_accounts = company_accounts(company_name, path_to_chromedriver)
				all_accounts = [company_name] + other_accounts
				
				tweets_final = pd.DataFrame({})
				for company_profile in all_accounts:
					
					# Twitter search URL for company
					urlpage = 'https://twitter.com/search?q=%40' + company_profile

					# Extract html data from twitter
					soup = BeautifulSoup(scroll_tweets(urlpage, path_to_chromedriver, int(50000)), "html.parser")
					
					# Extract text from html and clean
					tweets_list = twitter_page_extract(soup, company_profile)
					
					tweets_final = pd.concat([tweets_final, tweets_list], axis = 0)
					
					self.statusUpdateText = 'Finished extraction for {}\n'.format(company_profile)
					self.progressEvent.emit(self.statusUpdateText)
				
				data = self.queryDatabase(tweets_final)
				
# 				for company_profile in all_accounts:
# 					
# 					# Twitter search URL for company
# 					urlpage = 'https://twitter.com/search?q=%40' + company_profile
# 					
# 					self.statusUpdateText = 'Starting extraction for {} ...'.format(company_profile)
# 					self.progressEvent.emit(self.statusUpdateText)
# 					
# 					# Extract html data from twitter
# 					soup = BeautifulSoup(scroll_tweets(urlpage, self.path_to_chromedriver, int(self.bsLimit)), "html.parser")
# 					
# 					# Extract text from html and clean
# 					tweets_list = twitter_page_extract(soup, company_profile)
# 					
# 					tweets_final = pd.concat([tweets_final, tweets_list], axis = 0)
# 					
# 					self.statusUpdateText = 'Finished extraction for {}\n'.format(company_profile)
# 					self.progressEvent.emit(self.statusUpdateText)
# 				
# 				data = self.queryDatabase(tweets_final)
			else:
				self.statusUpdateText = 'SQL database contains data for {}: loading...'.format(self.company_name)
				self.progressEvent.emit(self.statusUpdateText)
				data = self.queryDatabase([])
				
			self.predictiveModel(data)
			time.sleep(0.2)
			self.running = False