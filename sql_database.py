#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:24:34 2019

@author: ggilmore
"""
import sqlite3
import pandas as pd
import os
import numpy as np

database_path = "/home/ggilmore/Documents/GitHub/ECE9603_project/gui/twitter_data.db"

if not os.path.exists(database_path):
	# Create table in db
	connection = sqlite3.connect(database_path)
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

# Determine ID number to be used
company_query = 'Bell'
data_path = r"/home/ggilmore/Documents/GitHub/ECE9603_project/example_datasets/twitter_bell.csv" 

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

cursor.execute("SELECT company FROM companies") 
result = cursor.fetchall()
if not any(x[0] == company_query for x in result):
	id_number = len(result)+1
	data = pd.read_csv(data_path, sep='\t')
	if 'account' not in list(data):
		data['account'] = np.repeat(company_query,len(data))
		twitter_handle = np.unique(data['account'])
	else:
		twitter_handle = np.unique(data['account'])
		
	sql_data = [id_number, company_query, twitter_handle]
	data['id'] = np.repeat(id_number, len(data))

	connection = sqlite3.connect(database_path)
	cursor = connection.cursor()

	format_str = """INSERT INTO companies (id, company, handles)
				    VALUES ("{id_number}","{company_name}", "{twitter_handles}");"""
	sql_command = format_str.format(id_number = sql_data[0], company_name=sql_data[1], twitter_handles=sql_data[2])
	cursor.execute(sql_command)

	# Insert whole DataFrame into MySQL
	data.to_sql('tweet_data', con = connection, if_exists = 'append', chunksize = 1000, index=False)

connection.commit()
connection.close()

