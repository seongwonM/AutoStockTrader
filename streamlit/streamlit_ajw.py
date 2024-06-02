from model_ajw import Stock, LSTMModel

def get_model_prediction(stock_code, current_hour_key):
    # current_hour_key 이전 10개 데이터 가져오기
    cursor.execute('SELECT * FROM price_info WHERE stock_code = ? AND time_key < ? ORDER BY time_key DESC LIMIT ?', (stock_code, current_hour_key, 5))
    rows = cursor.fetchall()

    if len(rows) < 5:
        return None  # 데이터가 충분하지 않으면 None 반환

    # 데이터를 DataFrame으로 변환
    df = pd.DataFrame(rows, columns=['Datetime', 'stock_code', 'High', 'Low', 'Open', 'Close', 'Volume'])

    stock=Stock(df)
    stock.preprocessing()
    stock.add_col()
    stock.scale_col(stock.df.columns[[3,0,1,2,5,4]])
    train_loader=stock.data_loader(5, 'train')
    valid_loader=stock.data_loader(5, 'valid')
    test_loader=stock.data_loader(5, 't')
    stock.create_model()
    stock.model.load_state_dict(torch.load('close.pth'))
    stock.train(train_loader, valid_loader, test_loader, 't', num_epoch = 100)
    predicted=stock.pred_value('t')

    return predicted