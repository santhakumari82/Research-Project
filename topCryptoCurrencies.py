# Filename: topCryptoCurrencies.py
# Description: This python program is used to fetch the top crypto currencies from coinmarketcap.com

# import required Python libraries
from requests import Session
import json
import pandas as pd
from pandas.io.json import json_normalize
from jproperties import Properties
# import user defined modules
from commonUtils import CommonUtils

# This class is used to fetch the top 20 cryptocurrency symbols from coinmarket api
class TopCryptoCurrencies:
    
    Coinmarket_API_URL = ''
    
    # Function to get the latest crypto currency data
    def get_cryptocurrency_symbols():
        configs = Properties()

        # Loading the property file to fetch keys
        with open('app-config.properties', 'rb') as config_file:
            configs.load(config_file)
            Coinmarket_API_URL = configs.get("COINMARKET_CAP_API").data
            API_KEY = configs.get("API_KEY").data
            DB_NAME = configs.get("DB_SCHEMA").data
        # API parameters required to fetch crypto currency data
        parameters = {'convert': 'CAD',
                      'start': '1',
                      'limit': '20'}

        headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': API_KEY
        }

        session = Session()
        session.headers.update(headers)
        # Fetching the top crypto symbols from coin market api
        response = session.get(Coinmarket_API_URL, params=parameters)
        print("Cryptocurrency Symbols retrieved from Coin Market API...")
        print(response.text)
        
        crypto_info = json.loads(response.text)['data']
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            for row in range(1,20):
                crypto_info = json.loads(response.text)['data']
                df = pd.json_normalize(crypto_info)
                
            data = response.json()
            crypto_list = []
            for number, row in enumerate(data['data']):
                crypto_list.append([row['symbol'],row['name']])
                
            df_pandas = pd.DataFrame(crypto_list, columns=["symbol","name"]) 
            # Get the DB Connection
            con = CommonUtils.getDBConnection()

            try:
                # Open the cursor
                with con.cursor() as cursor:
                    # Selecting the database
                    con.select_db(DB_NAME)
                    SQL_drop_table = "DROP TABLE IF EXISTS CRYPTO_TOP_CURRENCIES"
                    cursor.execute(SQL_drop_table)
                    # Creating the table for saving the crypto top currency symbols
                    SQL_create_table = "CREATE TABLE CRYPTO_TOP_CURRENCIES(ID INT UNSIGNED NOT NULL AUTO_INCREMENT,PRIMARY KEY(ID),SYMBOL CHAR(5), NAME VARCHAR(25))"
                    cursor.execute(SQL_create_table)
                    # Insert statement to save in the DB table
                    SQL_insert_data = "INSERT INTO CRYPTO_TOP_CURRENCIES(SYMBOL,NAME) VALUES(%s,%s)"
                    # Iterating through the data for saving in the database
                    for i, row in df_pandas.iterrows():
                        # print(tuple(row))
                        cursor.execute(SQL_insert_data, tuple(row))
                    print("Top Crypto Currency Symbols are inserted successfully")

            finally:
                # close the connection
                con.close()
    # call the method to get the crypto symbols            
    get_cryptocurrency_symbols()  

