# Filename: dailyCryptoScheduler.py
# Description: This python program is used to run the dailyCryptoData as a background scheduler task

import time

from apscheduler.schedulers.background import BackgroundScheduler
import os

# This method will schedule the daily run of the crypto currency data fetch
def cryptojob():
    os.system('dailyCryptoData.py')

if __name__ == '__main__':
    # creating the BackgroundScheduler object
    scheduler = BackgroundScheduler()
    # setting the scheduled task
    scheduler.add_job(cryptojob(), 'interval', hours=24)
    # starting the scheduled task using the scheduler object
    scheduler.start()
    try:
        # To simulate continous app activity (which keeps the main thread alive).
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):       
        scheduler.shutdown()
