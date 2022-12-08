# Filename: dailyCryptoData.py
# Description: This python program is used to fetch the daily data from yahoo finance

# import python required libraries
import yfinance as yf
import pandas as pd
import pymysql as pysql
import schedule
from datetime import date
import calendar

#import util functions
from commonUtils import CommonUtils

# This class is used to get daily crypto data
class DailyCryptoData:

    def getDailyCryptoData():
        current_day = date.today()
        print("Collecting daily data for ",current_day)
        # do not run the daily data for week ends.
        if (calendar.day_name[current_day.weekday()] != 'Saturday' and calendar.day_name[current_day.weekday()] != 'Sunday'):
            con = CommonUtils.getDBConnection()     
            try:
                with con.cursor() as cursor:
                    con.select_db("CRYPTO_CURRENCY")
                    SQL_select_data = "select CONCAT(SYMBOL,'-CAD') from CRYPTO_TOP_CURRENCIES"
                    cursor.execute(SQL_select_data)
                    crypto_symbol_list = cursor.fetchall()
                    merged_crypto_data = []
                    for row in crypto_symbol_list:                 
                        crypto_ticker = yf.Ticker(row[0])
                        crypto_data = crypto_ticker.history(period="1d")
                        #print(row[0])
                        #print(crypto_data)
                        # to avoid no data found error
                        if (row[0] != "BUSD-CAD" and row[0] != "UNI-CAD" and row[0] != "WBTC-CAD" and row[0] != "LEO-CAD"):
                            symbol = row[0].split("-")
                            crypto_data["Crypto_Symbol"] = symbol[0]                    
                            merged_crypto_data.append(crypto_data)
                            df_crypto = pd.concat(merged_crypto_data)
                    df_crypto = df_crypto.reset_index()

                    pd.set_option("display.max_rows", None, "display.max_columns", None)
                    df_crypto = df_crypto.iloc[1:, :]
                    print(df_crypto.head(5))
                    # load the data
                    SQL_insert_data = "INSERT INTO CRYPTO_CURRENCY_DATA(DATE,OPEN,HIGH,LOW,CLOSE,VOLUME,DIVIDENDS,STOCK_SPLITS,CRYPTO_SYMBOL) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    for i, row in df_crypto.iterrows():                        
                        cursor.execute(SQL_insert_data, tuple(row))
                    print("Daily data collection is successfull!")
            finally:
                    # closing the connection
                    con.close()

    if __name__ == "__main__":
        getDailyCryptoData()
