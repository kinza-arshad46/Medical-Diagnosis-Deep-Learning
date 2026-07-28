"""
==========================================================
Medical Diagnosis AI
Model Evaluation Pipeline
==========================================================

Author : Kinza Arshad

Description:
Evaluates trained model on the test dataset
and generates evaluation metrics.
==========================================================
"""

import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.utils.logger import logger
from src.utils.device import DeviceManager
from src.utils.custom_exception import (
    MedicalDiagnosisException
)

from src.models.model import (
    MedicalDiagnosisModel
)

from src.utils.checkpoint import (
    CheckpointManager
)

from src.pipeline.data_pipeline import (
    DataPipeline
)

from src.visualization.plots import (
    PlotGenerator
)

from src.reports.report_generator import (
    ReportGenerator
)


class ModelEvaluator:

    """
    Evaluate trained Medical Diagnosis Model.
    """

    def __init__(self):

        self.device = DeviceManager.get_device()

        logger.info(
            f"Evaluation Device : {self.device}"
        )

        self.model = (
            MedicalDiagnosisModel()
            .get_model()
            .to(self.device)
        )

        self.checkpoint = CheckpointManager()

        self.checkpoint.load_checkpoint(
            model=self.model,
            best=True
        )

        pipeline = DataPipeline()

        (
            _,
            _,
            self.test_loader,
            self.class_names

        ) = pipeline.get_dataloaders()

        self.plotter = PlotGenerator()

        self.report_generator = (
            ReportGenerator()
        )

        logger.info(
            "Evaluator initialized successfully."
        )

    def evaluate(self):

        """
        Evaluate complete test dataset.
        """

        try:

            self.model.eval()

            predictions = []

            labels = []

            probabilities = []

            image_paths = []

            with torch.no_grad():

                progress = tqdm(

                    self.test_loader,

                    desc="Evaluating"

                )

                for batch in progress:

                    images = batch["image"].to(
                        self.device
                    )

                    target = batch["label"].to(
                        self.device
                    )

                    outputs = self.model(images)

                    probs = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted = torch.max(
                        probs,
                        dim=1
                    )

                    predictions.extend(
                        predicted.cpu().numpy()
                    )

                    labels.extend(
                        target.cpu().numpy()
                    )

                    probabilities.extend(
                        confidence.cpu().numpy()
                    )

                    image_paths.extend(
                        batch["path"]
                    )

            logger.info(
                "Evaluation completed successfully."
            )

            return (

                labels,

                predictions,

                probabilities,

                image_paths

            )

        except Exception as e:

            raise MedicalDiagnosisException(e)
    def generate_reports(
        self,
        labels,
        predictions,
        probabilities,
        image_paths
    ):
        """
        Generate evaluation metrics,
        visualizations and reports.
        """

        try:

            accuracy = accuracy_score(
                labels,
                predictions
            )

            precision = precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0
            )

            try:

                roc_auc = roc_auc_score(
                    labels,
                    probabilities
                )

            except Exception:

                roc_auc = None

            report_text = classification_report(

                labels,

                predictions,

                target_names=self.class_names,

                zero_division=0

            )

            report_dict = classification_report(

                labels,

                predictions,

                target_names=self.class_names,

                output_dict=True,

                zero_division=0

            )

            metrics = {

                "Accuracy": round(
                    accuracy,
                    4
                ),

                "Precision": round(
                    precision,
                    4
                ),

                "Recall": round(
                    recall,
                    4
                ),

                "F1 Score": round(
                    f1,
                    4
                ),

                "ROC AUC":

                    round(roc_auc, 4)

                    if roc_auc is not None

                    else "Not Available"

            }

            logger.info(metrics)

            self.plotter.plot_confusion_matrix(

                labels,

                predictions,

                self.class_names

            )

            if roc_auc is not None:

                self.plotter.plot_roc_curve(

                    labels,

                    probabilities

                )

                self.plotter.plot_precision_recall_curve(

                    labels,

                    probabilities

                )

            self.report_generator.save_classification_report(

                report_text,

                report_dict

            )

            self.report_generator.save_prediction_results(

                image_paths,

                labels,

                predictions,

                probabilities

            )

            self.report_generator.save_evaluation_summary(

                metrics

            )

            logger.info(

                "All evaluation reports generated."

            )

            return metrics

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def run(self):
        """
        Complete evaluation pipeline.
        """

        try:

            logger.info(
                "=" * 70
            )

            logger.info(
                "Starting Model Evaluation..."
            )

            (
                labels,
                predictions,
                probabilities,
                image_paths

            ) = self.evaluate()

            metrics = self.generate_reports(

                labels,

                predictions,

                probabilities,

                image_paths

            )

            logger.info(
                "=" * 70
            )

            logger.info(
                "Evaluation Completed Successfully."
            )

            logger.info(
                "=" * 70
            )

            return metrics

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def print_summary(
        self,
        metrics
    ):

        logger.info("")

        logger.info("=" * 70)

        logger.info(
            "MODEL PERFORMANCE SUMMARY"
        )

        logger.info("=" * 70)

        for key, value in metrics.items():

            logger.info(

                f"{key:<15}: {value}"

            )

        logger.info("=" * 70)

        logger.info("")
        