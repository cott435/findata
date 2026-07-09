import pandas as pd

from findata import YahooFinance
from findata.database.ticker_sampling import TickerSampler
import time

tickers = list(TickerSampler()._tickers)

max_tickers = 40
infos = []

for i in range(0, len(tickers), max_tickers):
    subset = tickers[i:i + max_tickers]
    try:
        data = YahooFinance(subset).request_ticker_financials(start='1/1/2014')
        info, prices = data['info'], data['prices'].loc['daily']
    except Exception as e:
        print(e)
        time.sleep(61)
        data = YahooFinance(subset).request_ticker_financials(start='1/1/2014')
        info, prices = data['info'], data['prices'].loc['daily']
    infos.append(info)

final = pd.concat(infos)
final.to_excel('all_info.xlsx')

f = final[final['first_price_date']<pd.to_datetime('1/1/2018').date()]
f.to_excel('all_info_before_JAN2018.xlsx')

d = 1







