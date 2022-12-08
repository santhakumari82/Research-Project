# Filename: backTesting.py
# Description: This python program is used to backtesting historical data with optimization

from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pymysql as pysql
import pandas as pd
import talib
import seaborn as sns
import matplotlib.pyplot as plt

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

class BackTesting:
    def performBackTesting(self,crypto_data_df):
        symbol = input("Enter the Cryptocurrency symbol:")
        crypto_data = crypto_data_df.query("Crypto_Symbol == '"+symbol+"'")
        bt = Backtest(crypto_data
                      , BackTestingStrategy, cash = 10_000, commission = .002)
        stats = bt.run()  
        bt.plot()
        print(stats)
        
        lowerlimita,lowerlimitb = input("Enter the lower limit range:").split()
        upperlimita,upperlimitb = input("Enter the upper limit range:").split()
        
        
        stats, data = bt.optimize(upper_limit = range(int(upperlimita),int(upperlimitb),5),
                                  lower_limit = range(int(lowerlimita),int(lowerlimitb),5),
                                  rsi_window = range(10,30,2),
                                  maximize='Equity Final [$]',
                                  return_heatmap=True)
        print(stats)
        data_grouped = data.groupby(["upper_limit","lower_limit"]).mean().unstack()
        sns.heatmap(data_grouped, cmap="plasma")
        plt.show()
