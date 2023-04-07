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
    url = "https://www.latexlive.com/login##"
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
    username = browser.find_element_by_id("a_gotoreg_account")
    username.click()
    username = browser.find_element_by_id("input_regusername")
    username.clear()
    user = random_string(random.randint(10,20))
    username.send_keys(user)
    # username.send_keys("lenong0427@163.com")
    password = browser.find_element_by_xpath('//*[@id="input_regpassword"]')
    password.clear()
    password.send_keys("lovely520")
    password = browser.find_element_by_xpath('//*[@id="input_regpasswordagain"]')
    password.clear()
    password.send_keys("lovely520")
    time.sleep(24*60*60)

    browser.close()
    with open("username.txt","a") as f:
        f.write(user+"\n")









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
