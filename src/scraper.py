import json
import os
import re
import urllib
from typing import Dict, List

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager


def scrape() -> None:
    chrome_options = Options()
    chrome_options.add_experimental_option('detach', True)
    chrome_options.add_argument(f"user-data-dir={os.getenv('CHROME_DATA_PATH')}")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    url = 'https://www.polovniautomobili.com/auto-oglasi/pretraga?brand=&brand2=&price_from=&price_to=&year_from=&year_to' \
        '=&flywheel=&atest=&door_num=&submit_1=&with_images=1&date_limit=&showOldNew=all&modeltxt=&engine_volume_from' \
        '=&engine_volume_to=&power_from=&power_to=&mileage_from=&mileage_to=&emission_class=&seat_num=&wheel_side' \
        '=&registration=&country=&country_origin=&city=&registration_price=&page=&sort=renewDate_desc'
    driver.get(url)

    car_number = 0
    for i in range(40):
        cars = driver.find_elements(By.XPATH, "//article[contains(concat(' ', @class, ' '), ' classified ')]//div[@class='image']/child::a")
        for car in cars:
            car_number += 1
            car.send_keys(Keys.CONTROL + Keys.ENTER)
            driver.switch_to.window(driver.window_handles[1])

            data = extract_textual_data(driver)
            with open(f'../data/textual/{car_number}.json', 'w') as fp:
                json.dump(data, fp)

            extract_image_data(car_number, driver)

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        driver.find_element(By.XPATH, "//i[@class='uk-icon-angle-double-right']").click()


def extract_textual_data(driver: webdriver.Chrome) -> Dict:
    data = {}
    price = driver.find_element(By.XPATH, "//span[contains(@class, 'priceClassified')]").text
    data['cena'] = int(re.sub(r'[^0-9]+', '', price))

    general_info_keys = preprocess_textual_data(driver, "//*[contains(text(), 'Opšte informacije')]/parent::div//div[@class='uk-width-1-2']")
    general_info_values = preprocess_textual_data(driver, "//*[contains(text(), 'Opšte informacije')]/parent::div//div[@class='uk-width-1-2 uk-text-bold']")
    general_info = dict(zip(general_info_keys, general_info_values))
    data['opste informacije'] = general_info

    additional_info_keys = preprocess_textual_data(driver, "//*[contains(text(), 'Dodatne informacije')]/parent::div//div[@class='uk-width-1-2']")
    additional_info_values = preprocess_textual_data(driver, "//*[contains(text(), 'Dodatne informacije')]/parent::div//div[@class='uk-width-1-2 uk-text-bold']")
    additional_info = dict(zip(additional_info_keys, additional_info_values))
    data['dodatne informacije'] = additional_info

    data['sigurnost'] = preprocess_textual_data(driver, "//*[contains(text(), 'Sigurnost')]/parent::div//div[@class='uk-width-medium-1-4 uk-width-1-2 uk-margin-small-bottom']")
    data['oprema'] = preprocess_textual_data(driver, "//*[contains(text(), 'Oprema')]/parent::div//div[@class='uk-width-medium-1-4 uk-width-1-2 uk-margin-small-bottom']")
    data['stanje'] = preprocess_textual_data(driver, "//*[contains(text(), 'Stanje')]/parent::div//div[@class='uk-width-medium-1-4 uk-width-1-2 uk-margin-small-bottom']")
    try:
        description = driver.find_element(By.XPATH, "//*[contains(text(), 'Opis')]/parent::div//div[@class='uk-width-1-1 description-wrapper']")
        data['opis'] = replace_latin_letters(description.text.lower())
    except NoSuchElementException:
        data['opis'] = ''
    return data


def preprocess_textual_data(driver: webdriver.Chrome, xpath: str) -> List[str]:
    scraped_data = driver.find_elements(By.XPATH, xpath)
    return [replace_latin_letters(sd.text.lower()) for sd in scraped_data]


def extract_image_data(car_number: int, driver: webdriver.Chrome) -> None:
    images = driver.find_elements(By.XPATH, "//img[@class='js_counter']")
    images = [image.get_attribute('src') for image in images]
    images_path = f'../data/images/{car_number}'
    os.makedirs(images_path)
    image_number = 0
    for image in images:
        image_number += 1
        try:
            urllib.request.urlretrieve(image, f'{images_path}/{image_number}.png')
        except Exception:
            image_number -= 1


def replace_latin_letters(s: str) -> str:
    to_replace = {'č': 'c', 'ć': 'c', 'š': 's', 'đ': 'dj', 'ž': 'z', '\n': ' '}

    for char in to_replace.keys():
        s = s.replace(char, to_replace[char])

    return s


if __name__ == '__main__':
    scrape()

