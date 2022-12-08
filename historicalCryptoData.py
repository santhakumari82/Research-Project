# Filename: historicalCryptoData.py
# Description: This python program is used to get the historical crypto data from yahoo finance

# import required python libraries
import yfinance as yf
import pandas as pd
import pymysql as pysql

# import common utils module
from commonUtils import CommonUtils

# This class is used to fetch the historical data for the top 20 crypto currency symbols

class HistoricalCryptoData:

    # This method will convert a result set to list
    def convert_resultset_tolist(rs):
        crypto_list = []
        for element in rs:
            crypto_list.append(element)
        return crypto_list

    # This method will get the historical crypto data
    def getHistoricalCryptoData():
        # Get the DB Connection
        con = CommonUtils.getDBConnection()

        try:
            with con.cursor() as cursor:
                # Selecting the DATABASE Schema
                con.select_db("CRYPTO_CURRENCY")
                SQL_select_data = "select CONCAT(SYMBOL,'-CAD') from CRYPTO_TOP_CURRENCIES"
                cursor.execute(SQL_select_data)
                crypto_symbol_list = cursor.fetchall()
                merged_crypto_data = []
                # Iterating through the crypto symbols and fetching the cryptocurrency data from Yahoo Finance
                for row in crypto_symbol_list:                    
                    crypto_ticker = yf.Ticker(row[0])
                    crypto_data = crypto_ticker.history(period="max")
                    if (row[0] != "BUSD-GBP" and row[0] != "UNI-GBP" and row[0] != "WBTC-GBP" and row[0] != "LEO-GBP"):
                        symbol = row[0].split("-")                        
                        crypto_data["Crypto_Symbol"] = symbol[0]
                        #print(crypto_data)
                        merged_crypto_data.append(crypto_data)
                        df_crypto = pd.concat(merged_crypto_data)
            df_crypto = df_crypto.reset_index()
            pd.set_option("display.max_rows", None, "display.max_columns", None)
            #df_crypto = df_crypto.iloc[1:, :]
            df_crypto = df_crypto.iloc[1:, :9]
            print("data ready")
            print(df_crypto.head())
            with con.cursor() as cursor:
                SQL_drop_table = "DROP TABLE IF EXISTS CRYPTO_CURRENCY_DATA"
                cursor.execute(SQL_drop_table)
                SQL_create_table = "CREATE TABLE CRYPTO_CURRENCY_DATA(DATE DATE, OPEN FLOAT, HIGH FLOAT, LOW FLOAT, CLOSE FLOAT" \
                                   ",VOLUME LONG,DIVIDENDS INT,STOCK_SPLITS INT,CRYPTO_SYMBOL CHAR(10), CREATION_TIME DATETIME DEFAULT CURRENT_TIMESTAMP)"
                cursor.execute(SQL_create_table)
                # load the data
                SQL_insert_data = "INSERT INTO CRYPTO_CURRENCY_DATA(DATE,OPEN,HIGH,LOW,CLOSE,VOLUME,DIVIDENDS,STOCK_SPLITS,CRYPTO_SYMBOL) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                for i, row in df_crypto.iterrows():
                    cursor.execute(SQL_insert_data, tuple(row))
                print("Cryptocurrency Historical data fetching completed successfully")

        finally:
            con.close()

    if __name__ == '__main__':
        getHistoricalCryptoData()
