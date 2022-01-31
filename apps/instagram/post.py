from bs4 import BeautifulSoup
import pandas as pd

dataframe = pd.read_csv('hm.csv')

html = open('html2.txt', 'r', encoding='utf').read()
soup = BeautifulSoup(html)

for i in soup.find_all('div', {'class': 'C4VMK'}):
    user = i.find(class_='_6lAjh').find('a')['href']
    comment = i.find('span', {'class': ''}).text
    datetime = i.find('time')['datetime']

    dataframe = dataframe.append({
        'comment': comment,
        'user': user,
        'datetime': datetime
    }, ignore_index=True)

print(dataframe.shape)

dataframe.to_csv('hm.csv', index=False)
