import requests
from bs4 import BeautifulSoup
import schedule
import time

def get_stock_price():
    url = "https://finance.naver.com/item/sise.naver?code=005930"
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    price = soup.select_one("#_nowVal").text
    price = price.replace(',', "")
    price = int(price)
    print("현재 주식 가격:", price)
    return price

# 예약된 시간에 실행되는 함수
schedule.every().day.at("10:00").do(get_stock_price)  # 매일 10:00에 실행
schedule.every().day.at("11:00").do(get_stock_price)  # 매일 11:00에 실행
schedule.every().day.at("12:00").do(get_stock_price)  # 매일 12:00에 실행
schedule.every().day.at("13:00").do(get_stock_price)  # 매일 13:00에 실행
schedule.every().day.at("14:00").do(get_stock_price)  # 매일 14:00에 실행
schedule.every().day.at("15:00").do(get_stock_price)  # 매일 15:00에 실행

# 주기적으로 스케줄 확인 및 작업 실행
while True:
    schedule.run_pending()
    time.sleep(1)  # 스케줄 확인 주기

