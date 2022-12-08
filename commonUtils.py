# Filename: commonUtils.py
# Description: This python program is used as a common program to create MYSQL DB connection

import pymysql as pysql
from jproperties import Properties
import bcrypt as bp

class CommonUtils:

    def getDBConnection():
        configs = Properties()
        # load the properties file
        with open('app-config.properties', 'rb') as config_file:
            configs.load(config_file)
            db_host = configs.get("DB_HOST").data
            db_user = configs.get("DB_USER").data
            db_pwd = configs.get("DB_PWD").data            
            con = pysql.connect(host=db_host, user=db_user, password=db_pwd, autocommit=True)
            return con

   
