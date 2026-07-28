"""
==========================================================
Medical Diagnosis AI
Evaluation Metrics
==========================================================

Author : Kinza Arshad

Description:
Calculates classification metrics for training,
validation and testing.
==========================================================
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score
)

from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException


class Metrics:

    @staticmethod
    def calculate_metrics(y_true, y_pred):

        """
        Calculate standard classification metrics.
        """

        try:

            results = {

                "accuracy": accuracy_score(
                    y_true,
                    y_pred
                ),

                "precision": precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "recall": recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                ),

                "f1_score": f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

            }

            logger.info(
                "Evaluation metrics calculated successfully."
            )

            return results

        except Exception as e:

            raise MedicalDiagnosisException(e)

    @staticmethod
    def confusion(y_true, y_pred):

        """
        Compute confusion matrix.
        """

        try:

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            logger.info(
                "Confusion matrix generated."
            )

            return cm

        except Exception as e:

            raise MedicalDiagnosisException(e)

    @staticmethod
    def classification(y_true, y_pred):

        """
        Generate classification report.
        """

        try:

            report = classification_report(
                y_true,
                y_pred,
                digits=4
            )

            logger.info(
                "Classification report generated."
            )

            return report

        except Exception as e:

            raise MedicalDiagnosisException(e)

    @staticmethod
    def roc_auc(y_true, y_probability):

        """
        Calculate ROC-AUC Score.
        """

        try:

            score = roc_auc_score(
                y_true,
                y_probability
            )

            logger.info(
                f"ROC AUC Score: {score:.4f}"
            )

            return score

        except Exception as e:

            raise MedicalDiagnosisException(e)

    @staticmethod
    def pr_auc(y_true, y_probability):

        """
        Calculate Precision-Recall AUC.
        """

        try:

            score = average_precision_score(
                y_true,
                y_probability
            )

            logger.info(
                f"PR AUC Score: {score:.4f}"
            )

            return score

        except Exception as e:

            raise MedicalDiagnosisException(e)


if __name__ == "__main__":

    y_true = np.array([
        0, 1, 1, 0,
        1, 0, 1, 0
    ])

    y_pred = np.array([
        0, 1, 0, 0,
        1, 0, 1, 1
    ])

    y_probability = np.array([
        0.10,
        0.95,
        0.40,
        0.20,
        0.98,
        0.15,
        0.80,
        0.60
    ])

    metrics = Metrics.calculate_metrics(
        y_true,
        y_pred
    )

    print(metrics)

    print("\nConfusion Matrix\n")
    print(
        Metrics.confusion(
            y_true,
            y_pred
        )
    )

    print("\nClassification Report\n")
    print(
        Metrics.classification(
            y_true,
            y_pred
        )
    )

    print(
        "\nROC AUC:",
        Metrics.roc_auc(
            y_true,
            y_probability
        )
    )

    print(
        "\nPR AUC:",
        Metrics.pr_auc(
            y_true,
            y_probability
        )
    )