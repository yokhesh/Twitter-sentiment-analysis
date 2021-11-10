#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:40:59 2019

@author: ggilmore
"""
import re
import pandas as pd
import datetime
import sys
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
import numpy as np
from bs4 import BeautifulSoup

import os
import pandas as pd
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

#regex patterns
url_finder = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
problemchars = re.compile(r'[\[=\+/&<>;:!\\|*^\'"\?%$@)(_\,\.\t\r\n0-9-—\]]')

def company_accounts(company_name, path_to_chromedriver):
	chrome_options = Options()
	chrome_options.add_argument("--headless")
	
	browser = webdriver.Chrome(executable_path = path_to_chromedriver, options = chrome_options)
	browser.get('https://twitter.com/' + company_name)
	html = browser.page_source
	soup = BeautifulSoup(html, "html.parser")
	tweets = soup.find_all("p", {"class": "ProfileHeaderCard-bio"})
	if tweets:
		tweets = tweets[0].get_text().replace('\n','').split(' ')
		other_profiles = [re.sub('[\@\.]', '', x) for x in tweets if '@' in x and 'fr' not in x.lower()]
	else:
		other_profiles = []
		
	return other_profiles
		
def scroll_tweets(urlpage, path_to_chromedriver, limit = 50000):
	chrome_options = Options()
	chrome_options.add_argument("--headless")
	
	browser = webdriver.Chrome(executable_path = path_to_chromedriver, options = chrome_options)
	browser.get(urlpage)
	update_cnt = int(limit/10)
	oldHeight = 0
	while True:
		browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		time.sleep(3)
		newHeight = browser.execute_script("return document.body.scrollHeight")
		if np.any([newHeight > limit, newHeight==oldHeight]):
			print('100%')
			break
		if newHeight > update_cnt:
			print('{}%'.format(int(np.ceil((update_cnt/limit)*100))), end =" ")
			update_cnt += int(limit/10)
		oldHeight = newHeight
		
	html = browser.page_source
	browser.close()
	
	return html

def extract_text(i, company_name):
	user = (i.find('span', {'class':"username"}).get_text() if i.find('span', {'class':"username"}) is not None else "")
	link = ('https://twitter.com' + i.small.a['href'] if i.small is not None else "")
	date = (i.small.a['title'] if i.small is not None else "")
	text = (i.p.get_text().replace('\n','') if i.p is not None else "")

	tweet_date = datetime.datetime.strptime(date.split('-')[-1].strip(),"%d %b %Y")
	tweet_time = datetime.datetime.strptime(date.split('-')[0].strip(),"%I:%M %p")
	
	
	blog_dict = {
			"account": company_name,
			"url": link,
			"user": user,
			"date": tweet_date,
			"time": tweet_time.strftime('%H:%M'),
			#"blog_text": problemchars.sub(' ', url_finder.sub('', text))
            "blog_text": text
			}
	
	return blog_dict

def twitter_page_extract(soup, company_name):
	tweets_list = list()
	tweets = soup.find_all("li", {"data-item-type": "tweet"})
	for tweet in tweets:
		tweet_data = None
		try:
			tweet_data = extract_text(tweet, company_name)
		except:
			continue
	
		if tweet_data:
			if company_name.lower() not in tweet_data['user'].lower():
				tweets_list.append(tweet_data)
				sys.stdout.flush()
	
	tweets_list = pd.DataFrame(tweets_list)
	
	return tweets_list

path_to_chromedriver = os.path.join(r'D:\PhD\2sem\sentiment_analysis_paper\ECE9603_project-master\gui\chrome_driver', 'chromedriver.exe')
company_name = 'Netflix'
urlpage = 'https://twitter.com/search?q=%40' + company_name
urlpage = 'https://forum.anyscript.org/t/problems-with-lowerextremity-any-using-helen-hayes-marker-set/5464'
soup = BeautifulSoup(scroll_tweets(urlpage, path_to_chromedriver, int(50000)), "html.parser")
tweets_list = twitter_page_extract(soup, company_name)