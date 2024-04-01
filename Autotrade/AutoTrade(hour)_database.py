import requests
import json
import datetime
import time
import yaml
import sqlite3

# 데이터베이스 설정
conn = sqlite3.connect('stock_prices.db')
cursor = conn.cursor()

# price_info 테이블 생성
cursor.execute('''
CREATE TABLE IF NOT EXISTS price_info (
    time_key TEXT PRIMARY KEY,
    high INTEGER,
    low INTEGER,
    open INTEGER
)
''')
conn.commit()


with open('config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']
URL_BASE = _cfg['URL_BASE']

def send_message(msg):
    """디스코드 메세지 전송"""
    now = datetime.datetime.now()
    message = {"content": f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] {str(msg)}"}
    requests.post(DISCORD_WEBHOOK_URL, data=message)
    print(message)

def get_access_token():
    """토큰 발급"""
    headers = {"content-type":"application/json"}
    body = {"grant_type":"client_credentials",
    "appkey":APP_KEY, 
    "appsecret":APP_SECRET}
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    ACCESS_TOKEN = res.json()["access_token"]
    return ACCESS_TOKEN
    
def hashkey(datas):
    """암호화"""
    PATH = "uapi/hashkey"
    URL = f"{URL_BASE}/{PATH}"
    headers = {
    'content-Type' : 'application/json',
    'appKey' : APP_KEY,
    'appSecret' : APP_SECRET,
    }
    res = requests.post(URL, headers=headers, data=json.dumps(datas))
    hashkey = res.json()["HASH"]
    return hashkey

def get_current_price(code="005930"):
    """현재가 조회"""
    PATH = "uapi/domestic-stock/v1/quotations/inquire-price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json", 
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "appKey":APP_KEY,
            "appSecret":APP_SECRET,
            "tr_id":"FHKST01010100"}
    params = {
    "fid_cond_mrkt_div_code":"J",
    "fid_input_iscd":code,
    }
    res = requests.get(URL, headers=headers, params=params)
    # 현재가를 return해줌.
    return int(res.json()['output']['stck_prpr'])

def get_target_price():
    now = datetime.datetime.now()
    current_hour_key = now.strftime('%Y-%m-%d %H')
    previous_hour = now - datetime.timedelta(hours=1)
    previous_hour_key = previous_hour.strftime('%Y-%m-%d %H')

    # 현재 시간대의 시가
    if current_hour_key in price_info_dict:
        stck_oprc = price_info_dict[current_hour_key].open
    else:
        stck_oprc = None

    # 전 시간대의 고가와 저가
    if previous_hour_key in price_info_dict:
        stck_hgpr = price_info_dict[previous_hour_key].high
        stck_lwpr = price_info_dict[previous_hour_key].low
    else:
        stck_hgpr = None
        stck_lwpr = None

    if stck_oprc is not None and stck_hgpr is not None and stck_lwpr is not None:
        # 변동성 돌파 전략의 매수 목표가 계산
        target_price = stck_oprc + (stck_hgpr - stck_lwpr) * 0.5
        return target_price
    else:
        return None

class PriceInfo:
    def __init__(self):
        self.high = None
        self.low = None
        self.open = None

price_info_dict = {}

def update_price_info(current_price, current_time):
    time_key = current_time.strftime('%Y-%m-%d %H')

    if time_key not in price_info_dict:
        price_info_dict[time_key] = PriceInfo()
        price_info_dict[time_key].open = current_price  # 시간대 시작 시의 가격을 open으로 설정
        price_info_dict[time_key].high = current_price  # 초기 high 값을 현재 가격으로 설정
        price_info_dict[time_key].low = current_price   # 초기 low 값을 현재 가격으로 설정
        
        cursor.execute('''
        INSERT INTO price_info (time_key, high, low, open) VALUES (?, ?, ?, ?)
        ''', (time_key, current_price, current_price, current_price))
        conn.commit()
        
    else:
        # 이미 해당 시간대에 대한 정보가 존재하면, high와 low만 업데이트
        if current_price > price_info_dict[time_key].high:
            price_info_dict[time_key].high = current_price
        if current_price < price_info_dict[time_key].low:
            price_info_dict[time_key].low = current_price

        cursor.execute('''
        UPDATE price_info SET high = ?, low = ? WHERE time_key = ?
        ''', (price_info_dict[time_key].high, price_info_dict[time_key].low, time_key))
        conn.commit()


def get_hourly_price_info(hour):
    return price_info_dict.get(hour, None)

def get_stock_balance():
    """주식 잔고조회"""
    PATH = "uapi/domestic-stock/v1/trading/inquire-balance"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"TTTC8434R",
        "custtype":"P",
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "AFHR_FLPR_YN": "N",
        "OFL_YN": "",
        "INQR_DVSN": "02",
        "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N",
        "FNCG_AMT_AUTO_RDPT_YN": "N",
        "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "",
        "CTX_AREA_NK100": ""
    }
    res = requests.get(URL, headers=headers, params=params)
    stock_list = res.json()['output1']
    evaluation = res.json()['output2']
    stock_dict = {}
    send_message(f"====주식 보유잔고====")
    for stock in stock_list:
        if int(stock['hldg_qty']) > 0:
            stock_dict[stock['pdno']] = stock['hldg_qty']
            send_message(f"{stock['prdt_name']}({stock['pdno']}): {stock['hldg_qty']}주")
            time.sleep(0.1)
    send_message(f"주식 평가 금액: {evaluation[0]['scts_evlu_amt']}원")
    time.sleep(0.1)
    send_message(f"평가 손익 합계: {evaluation[0]['evlu_pfls_smtl_amt']}원")
    time.sleep(0.1)
    send_message(f"총 평가 금액: {evaluation[0]['tot_evlu_amt']}원")
    time.sleep(0.1)
    send_message(f"=================")
    return stock_dict

def get_balance():
    """현금 잔고조회"""
    PATH = "uapi/domestic-stock/v1/trading/inquire-psbl-order"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"TTTC8908R",
        "custtype":"P",
    }
    params = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": "005930",
        "ORD_UNPR": "65500",
        "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y",
        "OVRS_ICLD_YN": "Y"
    }
    res = requests.get(URL, headers=headers, params=params)
    cash = res.json()['output']['ord_psbl_cash']
    send_message(f"주문 가능 현금 잔고: {cash}원")
    return int(cash)

def buy(code="005930", qty="1"):
    """주식 시장가 매수"""  
    PATH = "uapi/domestic-stock/v1/trading/order-cash"
    URL = f"{URL_BASE}/{PATH}"
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code,
        "ORD_DVSN": "01",
        "ORD_QTY": str(int(qty)),
        "ORD_UNPR": "0",
    }
    headers = {"Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"TTTC0802U",
        "custtype":"P",
        "hashkey" : hashkey(data)
    }
    res = requests.post(URL, headers=headers, data=json.dumps(data))
    if res.json()['rt_cd'] == '0':
        send_message(f"[매수 성공]{str(res.json())}")
        return True
    else:
        send_message(f"[매수 실패]{str(res.json())}")
        return False

def sell(code="005930", qty="1"):
    """주식 시장가 매도"""
    PATH = "uapi/domestic-stock/v1/trading/order-cash"
    URL = f"{URL_BASE}/{PATH}"
    data = {
        "CANO": CANO,
        "ACNT_PRDT_CD": ACNT_PRDT_CD,
        "PDNO": code,
        "ORD_DVSN": "01",
        "ORD_QTY": qty,
        "ORD_UNPR": "0",
    }
    headers = {"Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"TTTC0801U",
        "custtype":"P",
        "hashkey" : hashkey(data)
    }
    res = requests.post(URL, headers=headers, data=json.dumps(data))
    if res.json()['rt_cd'] == '0':
        send_message(f"[매도 성공]{str(res.json())}")
        return True
    else:
        send_message(f"[매도 실패]{str(res.json())}")
        return False


# 자동매매 시작.
# 일괄매도는 필요없음.(일괄매도는 다음 시각.)
# 주식은 한 종목으로 가정
# 시간당 고가 저가를 알 수 있는 방법이 없음 --> 직접 수집해서 dict을 만들어야함.
# 9시에 대한 정보를 확인해야하므로 10시 부터 매매 가능

try:
    ACCESS_TOKEN = get_access_token()
    symbol = '005930'
    total_cash = get_balance()

    # 매수 매도 logic 에서 가장 중요한 부
    bought = False
    bought_time = None

    send_message('===국내 주식 자동매매 프로그램을 시작합니다===')

    while True:
        loop_start_time = datetime.datetime.now()

        t_now = datetime.datetime.now()
        t_start = t_now.replace(hour=10, minute=0, second=0, microsecond=0)
        t_sell = t_now.replace(hour=14, minute=50, second=0, microsecond=0)
        t_end = t_now.replace(hour=15, minute=0, second=0, microsecond=0)
        today = datetime.datetime.today().weekday()

        if today in [5,6]:  # 토요일이나 일요일이면 자동 종료
            send_message("주말이므로 프로그램을 종료합니다.")
            break
        
        if t_now >= t_end:
            send_message("오후 3시가 지났으므로 프로그램을 종료합니다.")
            break
        # 얘는 계속 업데이트 됨.
        current_price = get_current_price(symbol)
        update_price_info(current_price, t_now)
        
        # 매수 로직 (매수는 10시 부터 가능)
        if t_start < t_now < t_sell and not bought: # 매수 시간, 아직 매수하지 않았다면
            target_price = get_target_price()
            
            if target_price and target_price < current_price:
                buy_qty = int(total_cash // current_price)
                if buy_qty > 0:
                    result = buy(symbol , buy_qty)
                    if result:
                        bought = True
                        bought_time = t_now + datetime.timedelta(hours=1)  # 다음 시간 설정
                        send_message(f"{symbol} 매수 완료")
                        
        # 매도 로직
        if bought and bought_time.hour == t_now.hour and bought_time.minute <= t_now.minute:
            stoch_dict = get_stock_balance()
            qty = stock_dict.get(symbol , 0)
            if qty > 0:
                result = sell(symbol , qty)
                if result:
                    bought = False
                    send_message(f"{symbol} 매도완료")
          
        if t_now >= t_sell and bought:
            # 오후 2시 50분 이후, 아직 매도하지 않았다면 강제 매도
            stock_dict = get_stock_balance()  # 보유 주식 조회
            qty = stock_dict.get(symbol, 0)
            if qty > 0:
                sell(symbol, qty)
                bought = False
                send_message(f"장 마감 강제 매도: {symbol}")

        loop_end_time = datetime.datetime.now()
        elapsed_time = (loop_end_time - loop_start_time).total_seconds()
        sleep_time = max(60 - elapsed_time, 0)

        time.sleep(sleep_time)  # API 성능에 따라 변경

except Exception as e:
    send_message(f"[오류 발생]{e}")                  
            


finally:
    conn.close()
    send_message("프로그램이 종료되었습니다.")  
     