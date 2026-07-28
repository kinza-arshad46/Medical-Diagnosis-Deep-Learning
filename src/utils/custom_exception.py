"""
==========================================================
Medical Diagnosis AI
Custom Exception Handling
==========================================================

Author : Kinza Arshad

Description:
Custom exception class for centralized error handling.
Captures detailed debugging information including
file name, line number, and original error message.
==========================================================
"""

import sys
from pathlib import Path

from src.utils.logger import logger


class MedicalDiagnosisException(Exception):
    """
    Custom Exception for Medical Diagnosis AI Project.
    """

    def __init__(self, error_message, error_detail=sys):
        super().__init__(error_message)

        self.error_message = self._get_detailed_error(
            error_message,
            error_detail
        )

        logger.error(self.error_message)

    @staticmethod
    def _get_detailed_error(error_message, error_detail):
        """
        Create detailed error information.
        """

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is None:
            return error_message

        file_name = Path(exc_tb.tb_frame.f_code.co_filename).name

        line_number = exc_tb.tb_lineno

        function_name = exc_tb.tb_frame.f_code.co_name

        return (
            "\n"
            "========== Medical Diagnosis Exception ==========\n"
            f"File      : {file_name}\n"
            f"Function  : {function_name}\n"
            f"Line No   : {line_number}\n"
            f"Error     : {error_message}\n"
            "================================================="
        )

    def __str__(self):
        return self.error_message