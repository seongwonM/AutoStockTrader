import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 웹페이지 제목 설정
st.title('투자자 계좌 및 주식 관리 시스템')

# 사이드바 옵션
st.sidebar.header('사용자 설정')
investment_style = st.sidebar.selectbox('투자 성향 선택', ['보수적', '중립적', '공격적'])
st.sidebar.text('투자자의 성향에 따라\n추천 주식이 달라집니다.')

# 사용자 입력 폼 생성
with st.form("my_form"):
    st.write("### 계좌 정보 입력")
    account_number = st.text_input('계좌번호', '계좌번호를 여기에 입력하세요.')
    stock_code = st.text_input('주식코드', '주식코드를 여기에 입력하세요.')
    
    # 폼 제출 버튼
    submitted = st.form_submit_button("제출")
    if submitted:
        st.success("계좌 및 주식 정보가 성공적으로 제출되었습니다.")
        st.write(f"계좌번호: {account_number}")
        st.write(f"주식코드: {stock_code}")

# 포트폴리오 및 시장 요약
st.write("### 현재 포트폴리오")
portfolio = {
    "삼성전자": "5,000 주",
    "SK하이닉스": "2,000 주",
    "현대자동차": "1,000 주"
}
portfolio_df = st.table(portfolio)

st.write("### 주가 시장 상황")
market_summary = {
    "코스피": "2,450.25 상승",
    "코스닥": "980.88 하락",
    "다우존스": "34,200.67 상승"
}
market_df = st.table(market_summary)

# 차트 추가
st.write("### 주식 가격 차트")
# 임의의 데이터로 차트 생성
dates = pd.date_range(start='2023-01-01', periods=100)
prices = np.random.normal(100, 10, size=(100,))

plt.figure(figsize=(10, 5))
plt.plot(dates, prices, label='가격')
plt.title('주식 가격 변동')
plt.xlabel('날짜')
plt.ylabel('가격')
plt.legend()
st.pyplot(plt)

# 스타일링 및 레이아웃
st.markdown("""
<style>
.streamlit-container {
    padding-top: 5rem;
    padding-bottom: 5rem;
}
.header {
    color: blue;
}
</style>
""", unsafe_allow_html=True)
