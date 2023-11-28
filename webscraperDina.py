from bs4 import BeautifulSoup
from selenium import webdriver
import requests
import time
import os
import base64
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService

leaves = ["Maple", "Birch", "Oak", "Fraxinus", "Ilex", "Salix fragilis"]

def get_img_urls(keyword):
    webdriver_path = "./chromedriver.exe"
    webdriver_service = ChromeService(executable_path=webdriver_path)
    driver = webdriver.Chrome(service=webdriver_service)

    driver.get(f"https://www.google.com/search?q={keyword} leaf&tbm=isch")

    # Handle cookie consent pop-up
    try:
        accept_all_button = driver.find_element(By.XPATH, "//button[contains(., 'Accept all')]")
        accept_all_button.click()
    except:
        pass  # If there's no cookie consent pop-up, this will just continue

    for i in range(3):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(0.5)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    driver.quit()
    img_tags = soup.findAll('img', class_='rg_i Q4LuWd')
    img_urls = []
    for img_tag in img_tags:
        if img_tag.has_attr('data-src'):
            img_urls.append(img_tag['data-src'])
    return list(img_urls)

def download_image(url, keyword, index):
    if url.startswith('data:image;base64'):
        # Extract base64 data from the URL and save it as an image
        base64_data = url.split(',')[1]
        img_data = base64.b64decode(base64_data)
    else:
        response = requests.get(url)
        img_data = response.content

    file = open(f"./datasetDina/{keyword}/{index+1}.jpg", "wb")
    file.write(img_data)
    file.close()

def make_keyword_dir(keyword):
    cwd = os.getcwd()
    leaves_images_dir = os.path.join(cwd, "datasetDina")
    if not os.path.exists(leaves_images_dir):
        os.mkdir(leaves_images_dir)
    bird_dir = os.path.join(leaves_images_dir, keyword)
    if not os.path.exists(bird_dir):
        os.mkdir(bird_dir)

for keyword in leaves:
    make_keyword_dir(keyword)
    img_urls = get_img_urls(keyword)
    for i in range(min(300, len(img_urls))):
        download_image(img_urls[i], keyword, i)
