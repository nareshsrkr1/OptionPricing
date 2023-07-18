import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Placeholder for the log file name
log_filename = 'logs/model_app.log'

# Placeholder for the log handler
logger_handler = None


def set_log_filename(filename):
    global log_filename
    log_filename = filename


def initialize_log_handler():
    global logger_handler, log_filename
    if logger_handler:
        logger.removeHandler(logger_handler)

    logger_handler = logging.FileHandler(log_filename)
    logger_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
