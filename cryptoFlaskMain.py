# Filename: cryptoFlaskMain.py
# Description: This python program is used to build the flask framework web page for backtest crypto symbols
# How to run? : Run this python program as an individual file and type http://127.0.0.1:5000/ in browser to view the flask backtesting main page

from flask import Flask, render_template, url_for, request
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pymysql as pysql
import pandas as pd
import talib
import seaborn as sns
import matplotlib.pyplot as plt

# user defined
from backTesting import BackTesting
from commonUtils import CommonUtils
           
app = Flask(__name__)

@app.route("/")
def mainPage():
    return render_template('index.html')

@app.route("/", methods=['GET', 'POST'])
def result():
  QTc_result = False
  if request.method == 'POST':
    form = request.form
    QTc_result = calculate_backtesting(form)
  return render_template('result.html', QTc_result=QTc_result)
  

@app.route("/about_link")
def about_link():
    return render_template('About.html')

@app.route("/learnmore_link")
def learnmore_link():
    return render_template('LearnMore.html')

@app.route("/backtesting_link")
def backtesting_link():
    return render_template('Page-1.html')

class BackTestingStrategy(Strategy):
    upper_limit = 70
    lower_limit = 45
    rsi_window = 14
    
    def init(self):
        self.rsi = self.I(talib.RSI,self.data.Close, 14)
    def next(self):
        if crossover(self.rsi, self.upper_limit):
            self.position.close()
        elif (crossover(self.lower_limit, self.rsi)):
            self.buy()
            
def calculate_backtesting(form):
    con = CommonUtils.getDBConnection()
    try:
        with con.cursor() as cursor:
            con.select_db("CRYPTO_CURRENCY")
            SQL_select_data = "select * from CRYPTO_CURRENCY_DATA"
            cursor.execute(SQL_select_data)
            crypto_data = cursor.fetchall()
            crypto_data_df = pd.DataFrame(crypto_data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits', 'Crypto_Symbol', 'Creation_Time'])
            cryptoSymbol = request.form['cryptoSymbol']
            crypto_data_df = crypto_data_df.set_index('Date')
            trend_data = crypto_data_df.query("Crypto_Symbol == '"+cryptoSymbol+"'")
            bt = Backtest(trend_data, BackTestingStrategy, cash = 10_000, commission = .002)
            stats = bt.run()
            bt.plot()            
            SQL_select_data = "select CONCAT_WS('-',SYMBOL,NAME) from CRYPTO_TOP_CURRENCIES WHERE SYMBOL='"+cryptoSymbol+"'"
            cursor.execute(SQL_select_data)
            symbol_desc = cursor.fetchall()
            symbol = ''
            for row in symbol_desc:
                symbol = row[0]
            return (stats,symbol)
    finally:
        con.close()

  
if __name__ == '__main__':
    app.run(debug=True)


