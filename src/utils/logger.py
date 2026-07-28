"""
==========================================================
Medical Diagnosis AI
Logging Utility
==========================================================
Author : Kinza Arshad

Description:
Centralized logging system for the project.
==========================================================
"""

import logging
from pathlib import Path
from datetime import datetime

from src.utils.config_loader import ConfigLoader


class Logger:
    """
    Creates and manages project logging.
    """

    def __init__(self):

        config = ConfigLoader()

        log_directory = Path(config.get("paths", "logs_dir"))

        log_directory.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_file = log_directory / f"medical_ai_{timestamp}.log"

        self.logger = logging.getLogger("MedicalDiagnosisAI")

        self.logger.setLevel(
            getattr(logging, config.get("logging", "level", default="INFO"))
        )

        if not self.logger.handlers:

            formatter = logging.Formatter(
                "[%(asctime)s] | %(levelname)s | %(filename)s | "
                "%(funcName)s | Line:%(lineno)d | %(message)s"
            )

            file_handler = logging.FileHandler(
                self.log_file,
                encoding="utf-8"
            )

            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()

            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger


logger = Logger().get_logger()


if __name__ == "__main__":

    logger.info("Project Started Successfully.")

    logger.warning("This is a warning message.")

    logger.error("This is a sample error.")

    logger.critical("Critical issue example.")