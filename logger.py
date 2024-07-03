import logging
import os
from datetime import datetime

# Get current timestamp
timestamp = f'{str(datetime.now().replace(microsecond=0))}'.replace(
    ' ', '_').replace(':', '_')

# Create log file name with timestamp
log_dir = 'logs'.replace('''\\''', '''/''')
os.makedirs(log_dir, exist_ok=True)
log_filename = f'{log_dir}/model_logs_{timestamp}.log'

# Set up logger
logger = logging.getLogger('custom_logger')
logger.setLevel(logging.DEBUG)

# Create file handler which logs messages
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Create console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter and add it to the handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)