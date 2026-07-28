"""
==========================================================
Medical Diagnosis AI
Report Generator
==========================================================

Author : Kinza Arshad

Description:
Generates professional evaluation reports.
==========================================================
"""

from pathlib import Path
import json

import pandas as pd

from src.utils.logger import logger
from src.utils.custom_exception import (
    MedicalDiagnosisException
)


class ReportGenerator:

    def __init__(self, output_directory="outputs/reports"):

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def save_classification_report(
        self,
        report_text,
        report_dictionary
    ):

        try:

            txt_path = (
                self.output_directory /
                "classification_report.txt"
            )

            with open(
                txt_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(report_text)

            csv_path = (
                self.output_directory /
                "classification_report.csv"
            )

            dataframe = pd.DataFrame(
                report_dictionary
            ).transpose()

            dataframe.to_csv(
                csv_path,
                index=True
            )

            logger.info(
                "Classification report saved."
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def save_prediction_results(
        self,
        image_paths,
        actual_labels,
        predicted_labels,
        confidence_scores
    ):

        try:

            dataframe = pd.DataFrame({

                "Image": image_paths,

                "Actual": actual_labels,

                "Predicted": predicted_labels,

                "Confidence": confidence_scores

            })

            save_path = (

                self.output_directory /

                "prediction_results.csv"

            )

            dataframe.to_csv(

                save_path,

                index=False

            )

            logger.info(

                "Prediction results saved."

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def save_training_history(
        self,
        history
    ):

        try:

            save_path = (

                self.output_directory /

                "training_history.json"

            )

            with open(

                save_path,

                "w",

                encoding="utf-8"

            ) as file:

                json.dump(

                    history,

                    file,

                    indent=4

                )

            logger.info(

                "Training history saved."

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def save_model_information(
        self,
        information
    ):

        try:

            save_path = (

                self.output_directory /

                "model_information.json"

            )

            with open(

                save_path,

                "w",

                encoding="utf-8"

            ) as file:

                json.dump(

                    information,

                    file,

                    indent=4

                )

            logger.info(

                "Model information saved."

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def save_evaluation_summary(
        self,
        metrics
    ):

        try:

            save_path = (

                self.output_directory /

                "evaluation_summary.txt"

            )

            with open(

                save_path,

                "w",

                encoding="utf-8"

            ) as file:

                file.write(

                    "========== MODEL EVALUATION ==========\n\n"

                )

                for key, value in metrics.items():

                    file.write(

                        f"{key}: {value}\n"

                    )

            logger.info(

                "Evaluation summary saved."

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    generator = ReportGenerator()

    generator.save_training_history({

        "train_loss": [0.80, 0.55],

        "validation_loss": [0.82, 0.60]

    })

    print("Reports generated successfully.")