import requests
from bs4 import BeautifulSoup
import sqlite3
import re

def scrape(page_num, brand, car_list):
    url = f'https://www.autoscout24.nl/lst/{brand}?atype=C&cy=NL&desc=0&page={page_num}&search_id=5ieb69blcl&sort=age&source=listpage_pagination&ustate=N%2CU'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    articles = soup.findAll('article',
                            class_='cldt-summary-full-item listing-impressions-tracking list-page-item ListItem_article__qyYw7')

    if len(articles) < 2:
        return True

    pattern = r"-(\w+)-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"

    for article in articles:
        model = article.get('data-model')
        mileage = article.get('data-mileage')
        first_registration = article.get('data-first-registration')

        if mileage == 'unknown' or model == 'unknown':
            continue

        if len(first_registration) >= 4:
            first_registration = first_registration[-4:]
        elif first_registration == 'new':
            first_registration = 2024
        else:
            continue

        link_tag = article.find('a', class_='ListItem_title__ndA4s ListItem_title_new_design__QIU2b Link_link__Ajn7I')
        link = link_tag['href'] if link_tag else None
        if link is None:
            continue

        link = 'https://www.autoscout24.nl' + link.strip()
        match = re.search(pattern, link)
        if not match:
            continue
        color = match.group(1)
        car_data = {
            'first_registration': first_registration,
            'make': article.get('data-make'),
            'mileage': article.get('data-mileage'),
            'model': article.get('data-model'),
            'price': article.get('data-price'),
            'fuel_type': article.get('data-fuel-type'),
            'link': link,
            'color': color
        }

        car_list.append(car_data)

    return False

def link_exists(cursor, link):
    """Check if the link already exists in the database."""
    cursor.execute("SELECT 1 FROM cars WHERE link = ?", (link,))
    return cursor.fetchone() is not None

def save_to_db(car_list, db_name="cars.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS cars (
                        first_registration TEXT,
                        make TEXT,
                        mileage TEXT,
                        model TEXT,
                        price TEXT,
                        fuel_type TEXT,
                        color TEXT,
                        link TEXT UNIQUE
                    )''')

    for car in car_list:
        if not link_exists(cursor, car['link']):
            cursor.execute('''INSERT INTO cars 
                              (first_registration, make, mileage, model, price, fuel_type, color, link)
                              VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                              (car['first_registration'], car['make'], car['mileage'], car['model'],
                               car['price'], car['fuel_type'], car['color'], car['link']))

    conn.commit()
    conn.close()

    print(f"Data written to {db_name}")

brands = []


def main(brands):
    car_list = []
    for brand in brands:
        print(f'---Now analyzing: {brand}---')
        page_num = 1
        while True:
            no_more_pages = scrape(page_num, brand, car_list)
            page_num += 1
            if no_more_pages:
                print(f"No more pages after page {page_num - 1}")
                break
    save_to_db(car_list)

if __name__ == "__main__":
    main(brands)
