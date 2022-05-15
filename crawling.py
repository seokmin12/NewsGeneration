from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from tqdm import tqdm
import re

driver = webdriver.Chrome('/Users/seokmin/Desktop/project/word_machine_learning/chromedriver')

start_date = pd.to_datetime('2004-05-10')
end_date = pd.to_datetime('2022-05-14')

dates = pd.date_range(start_date, end_date, freq='D')

title_list = []

for date in tqdm(dates):
    date = str(date.to_pydatetime().strftime('%Y%m%d'))

    url = f'https://sports.news.naver.com/wfootball/news/index?page=1&date={date}&isphoto=N'

    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    news_list = soup.select_one("#_newsList")
    titles = news_list.findAll("a", class_="title")

    for title in titles:
        title = title.select_one('span').text
        title = re.sub(r"[^\.\?\!\w\d\s.]", "", title)
        title_list.append(title)

driver.quit()

title_list_set = set(title_list)
title_list = list(title_list_set)

raw = {'title': title_list}
data = pd.DataFrame(raw)
data.to_csv('/Users/seokmin/Desktop/project/word_machine_learning/data/news_data.csv', index=False)
