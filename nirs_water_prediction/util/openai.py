import datetime
import random
import time

import requests
from selenium.webdriver.chrome.options import Options

headers = {'Content-Type': "application/x-www-form-urlencoded", 'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) '
                                                                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                                                                              'Chrome/100.0.4692.36 Safari/537.36'}

import random
import string

def random_string(length):
    letters = string.ascii_lowercase + string.ascii_uppercase
    return ''.join(random.choice(letters) for i in range(length))
def daka():
    url = "https://chat1.aifks001.site/#/login/register?code=lenong1"
    from selenium import webdriver
    from selenium.webdriver.support.wait import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.by import By
    from bs4 import BeautifulSoup
    import lxml
    import os

    # PATH = os.environ['PATH']
    # PATH = "F:\pythonProject\pythonProject1-master\python学习\chromedriver.exe:" + PATH
    # os.environ["PATH"] = PATH

    # print(PATH)
    # os.system("unset http_proxy")
    # os.system("unset http_proxys")

    chrome_options = Options()

    # chrome_options.add_argument('--headless')

    chrome_options.add_argument('--disable-gpu')

    chrome_options.add_argument("window-size=1024,768")

    # chrome_options.add_argument("--no-sandbox")
    path = os.path.realpath(__file__)
    path = os.path.split(path)[0] + os.path.sep

    from selenium import webdriver
    from webdriver_manager.chrome import ChromeDriverManager

    browser = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)

    # driver = webdriver.Chrome(ChromeDriverManager().install())
    # browser = webdriver.Chrome()
    text_notice_text = "获取失败"
    all_liu = None
    remain_liu = None
    remain_day = None

    browser.get(url)

    username = browser.find_element_by_xpath('//*[@id="app"]/div/div[1]/div/div[2]/div/div/main/div/form/div[1]/div[1]/div/div[1]/div/input')
    username.clear()
    user = random_string(random.randint(10,20))
    username.send_keys(user)
    # username.send_keys("lenong0427@163.com")
    email = browser.find_element_by_xpath('//*[@id="app"]/div/div[1]/div/div[2]/div/div/main/div/form/div[2]/div[1]/div/div[1]/div/input')
    email.clear()

    email.send_keys( ''.join(random.choice("1234567890")  for i in range(10))+"@qq.com")
    password = browser.find_element_by_xpath('//*[@id="app"]/div/div[1]/div/div[2]/div/div/main/div/form/div[3]/div[1]/div/div[1]/div[1]/input')
    password.clear()
    password.send_keys("lovely520")
    password = browser.find_element_by_xpath('//*[@id="app"]/div/div[1]/div/div[2]/div/div/main/div/form/div[4]/div[1]/div/div[1]/div[1]/input')
    password.clear()
    password.send_keys("lovely520")

    f = browser.find_element_by_xpath('//*[@id="app"]/div/div[1]/div/div[2]/div/div/main/div/form/div[7]/div[1]/div/div/div/div/div[2]')
    f.click()

    with open("username.txt","a") as f:
        f.write(user+"\n")

    try:

        wait = WebDriverWait(browser,24 * 60*60, 5)
        wait.until_not(lambda driver: driver.current_url)

    finally:
        if browser is not  None:
            browser.close()










if __name__ == '__main__':
    while True:
        daka()
    #  try:
    #      setCircleTime()
    #  except Exception as e:
    #      print(e)
    #      import pathlib
    #      f = pathlib.Path("/home/lenong0427/sh/sockboom.txt")
    #      with open(f,"w") as file:
    #          file.write(str(e) +"\n" + str( datetime.datetime.now()))
