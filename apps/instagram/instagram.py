from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd
from bs4 import BeautifulSoup

PATH = '/Users/chipanatica/PycharmProjects/TFM/config/chromedriver'
driver = webdriver.Chrome(PATH)

driver.get("https://www.instagram.com/")

time.sleep(5)
username = driver.find_element(by=By.NAME, value='username')
password = driver.find_element(by=By.NAME, value="password")
username.clear()
password.clear()
username.send_keys("jose_chipana_t@hotmail.com")
password.send_keys("#onepiece13101997#")

login = driver.find_element_by_css_selector("button[type='submit']").click()

# save your login info?
time.sleep(10)
notnow = driver.find_element_by_xpath("//button[contains(text(), 'Ahora no')]").click()
# turn on notif
time.sleep(10)
notnow2 = driver.find_element_by_xpath("//button[contains(text(), 'Ahora no')]").click()

# searchbox
time.sleep(5)
searchbox = driver.find_element_by_css_selector("input[placeholder='Buscar']")
searchbox.clear()
searchbox.send_keys("zara")
time.sleep(5)
searchbox.send_keys(Keys.ENTER)
time.sleep(5)
searchbox.send_keys(Keys.ENTER)

# scroll
for i in range(20):
    scrolldown = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var scrolldown=document.body.scrollHeight;return scrolldown;")
    time.sleep(5)

posts = []
links = driver.find_elements(by=By.TAG_NAME, value='a')
for link in links:
    post = link.get_attribute('href')
    if '/p/' in post:
        posts.append(post)

df_post = pd.DataFrame(posts, columns=['link'])
df_post.to_csv('zara_post.csv', index=False)

comment_list = []

download_url = ''
for post in posts:
    driver.get(post)
    time.sleep(5)

    for i in range(10):
        driver.find_element(by=By.CLASS_NAME, value='qF0y9').click()
        time.sleep(2)

    soup = BeautifulSoup(driver.page_source)

    comments = soup.find_all('ul', {'class': 'Mr508'})
    for comment in comments:
        user = comment.find('a')['href']
        comm = comment.find_all('div', {'class': 'C4VMK'})[0]  # .find('span')
        print(user)
        print(comm.text)

        comment_list.append([user, comm])
