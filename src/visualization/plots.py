"""
==========================================================
Medical Diagnosis AI
Visualization Module
==========================================================

Author : Kinza Arshad

Description:
Generates professional visualizations for
training and model evaluation.
==========================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    confusion_matrix
)

from src.utils.logger import logger
from src.utils.custom_exception import (
    MedicalDiagnosisException
)


class PlotGenerator:

    def __init__(self, output_directory="outputs/plots"):

        self.output_directory = Path(output_directory)

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )

    def plot_training_history(self, history):

        try:

            epochs = range(
                1,
                len(history["train_loss"]) + 1
            )

            plt.figure(figsize=(8,5))

            plt.plot(
                epochs,
                history["train_loss"],
                label="Train Loss"
            )

            plt.plot(
                epochs,
                history["validation_loss"],
                label="Validation Loss"
            )

            plt.xlabel("Epoch")

            plt.ylabel("Loss")

            plt.title("Training Loss")

            plt.legend()

            plt.grid(True)

            plt.tight_layout()

            save_path = (
                self.output_directory /
                "training_loss.png"
            )

            plt.savefig(save_path)

            plt.close()

            plt.figure(figsize=(8,5))

            plt.plot(
                epochs,
                history["train_accuracy"],
                label="Train Accuracy"
            )

            plt.plot(
                epochs,
                history["validation_accuracy"],
                label="Validation Accuracy"
            )

            plt.xlabel("Epoch")

            plt.ylabel("Accuracy")

            plt.title("Training Accuracy")

            plt.legend()

            plt.grid(True)

            plt.tight_layout()

            save_path = (
                self.output_directory /
                "training_accuracy.png"
            )

            plt.savefig(save_path)

            plt.close()

            logger.info(
                "Training history plots saved."
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def plot_confusion_matrix(
        self,
        y_true,
        y_pred,
        class_names
    ):

        try:

            cm = confusion_matrix(
                y_true,
                y_pred
            )

            plt.figure(figsize=(6,6))

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=class_names
            )

            display.plot()

            plt.tight_layout()

            save_path = (
                self.output_directory /
                "confusion_matrix.png"
            )

            plt.savefig(save_path)

            plt.close()

            logger.info(
                "Confusion Matrix saved."
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def plot_roc_curve(
        self,
        y_true,
        probabilities
    ):

        try:

            plt.figure(figsize=(6,6))

            RocCurveDisplay.from_predictions(
                y_true,
                probabilities
            )

            plt.tight_layout()

            save_path = (
                self.output_directory /
                "roc_curve.png"
            )

            plt.savefig(save_path)

            plt.close()

            logger.info(
                "ROC Curve saved."
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def plot_precision_recall_curve(
        self,
        y_true,
        probabilities
    ):

        try:

            plt.figure(figsize=(6,6))

            PrecisionRecallDisplay.from_predictions(
                y_true,
                probabilities
            )

            plt.tight_layout()

            save_path = (
                self.output_directory /
                "precision_recall_curve.png"
            )

            plt.savefig(save_path)

            plt.close()

            logger.info(
                "Precision Recall Curve saved."
            )

        except Exception as e:

            raise MedicalDiagnosisException(e)