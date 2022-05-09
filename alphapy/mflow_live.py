"""
Run a live AlphaPy model.
"""


#
# Imports
#

from alphapy.alphapy_main import prediction_pipeline
from alphapy.globals import PSEP
from alphapy.mflow_main import get_market_config
from alphapy.model import get_model_config
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio
from datetime import datetime
import logging
import os


#
# Initialize logger
#

logger = logging.getLogger(__name__)


#
# Market Data Loop
#

def tick():
    print('Tick! The time is: %s' % datetime.now())


#
# Main Program
#

if __name__ == '__main__':

    # Initialize Logger

    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                        filename="market_live.log", filemode='a', level=logging.INFO,
                        datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                                  datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # Read model configuration file
    _, specs = get_model_config()

    # Read stock configuration file

    _, market_specs = get_market_config(alphapy_specs)
    logger.info(market_specs)

    # Initialize market prediction pipeline

    prediction_pipeline(model)
    logger.info(market_specs)

    # Initialize Scheduler

    scheduler = AsyncIOScheduler()
    scheduler.add_job(tick, 'interval', seconds=3)
    scheduler.start()
    print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))

    # Execution will block here until Ctrl+C (Ctrl+Break on Windows) is pressed.

    try:
        asyncio.get_event_loop().run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass