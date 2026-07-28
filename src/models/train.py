"""
==========================================================
Medical Diagnosis AI
Training Pipeline
==========================================================

Author : Kinza Arshad

Description:
Production-ready Training Pipeline
for Medical Image Classification.
==========================================================
"""

import os
import time

import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.utils.logger import logger
from src.utils.seed import SeedManager
from src.utils.device import DeviceManager
from src.utils.checkpoint import CheckpointManager

from src.models.model import MedicalDiagnosisModel
from src.models.loss import LossFunction
from src.models.optimizer import OptimizerBuilder
from src.models.scheduler import SchedulerBuilder
from src.models.metrics import Metrics

from src.data.dataset import DatasetManager
from src.data.augmentation import ImageAugmentation

from src.utils.config_loader import ConfigLoader
from src.utils.custom_exception import MedicalDiagnosisException


class Trainer:

    def __init__(self):

        self.config = ConfigLoader()

        SeedManager.set_seed(42)

        self.device = DeviceManager.get_device()

        logger.info(f"Training Device : {self.device}")

        self.epochs = self.config.get(
            "training",
            "epochs"
        )

        self.model = (
            MedicalDiagnosisModel()
            .get_model()
            .to(self.device)
        )

        self.criterion = (
            LossFunction()
            .get_loss()
            .to(self.device)
        )

        self.optimizer = (
            OptimizerBuilder()
            .get_optimizer(self.model)
        )

        self.scheduler = (
            SchedulerBuilder()
            .get_scheduler(self.optimizer)
        )

        self.checkpoint_manager = (
            CheckpointManager()
        )

        augmentation = ImageAugmentation()

        dataset_manager = DatasetManager()

        (
            self.train_loader,
            self.validation_loader,
            self.test_loader,
            self.class_names

        ) = dataset_manager.get_dataloaders(

            augmentation.train_augmentation(),

            augmentation.validation_augmentation(),

            augmentation.test_augmentation()

        )

        self.scaler = GradScaler()

        self.best_accuracy = 0.0

        self.best_loss = float("inf")

        self.early_stop_counter = 0

        self.history = {

            "train_loss": [],

            "validation_loss": [],

            "train_accuracy": [],

            "validation_accuracy": []

        }

        logger.info(
            "Trainer initialized successfully."
        )

    def train_one_epoch(self, epoch):

        """
        Train model for one epoch.
        """

        self.model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        progress_bar = tqdm(

            self.train_loader,

            desc=f"Epoch {epoch+1}/{self.epochs}",

            leave=False

        )

        for batch in progress_bar:

            images = batch["image"].to(self.device)

            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            with autocast():

                outputs = self.model(images)

                loss = self.criterion(

                    outputs,

                    labels

                )

            self.scaler.scale(loss).backward()

            self.scaler.step(

                self.optimizer

            )

            self.scaler.update()

            running_loss += loss.item()

            _, predictions = torch.max(

                outputs,

                dim=1

            )

            total += labels.size(0)

            correct += (

                predictions == labels

            ).sum().item()

            accuracy = 100 * correct / total

            progress_bar.set_postfix({

                "Loss":

                f"{loss.item():.4f}",

                "Accuracy":

                f"{accuracy:.2f}%"

            })

        epoch_loss = (

            running_loss /

            len(self.train_loader)

        )

        epoch_accuracy = (

            correct /

            total

        )

        logger.info(

            f"Epoch {epoch+1}"

            f" Train Loss : {epoch_loss:.4f}"

            f" Accuracy : {epoch_accuracy:.4f}"

        )

        return (

            epoch_loss,

            epoch_accuracy

        )
    def validate_one_epoch(self, epoch):
        """
        Validate model for one epoch.
        """

        self.model.eval()

        running_loss = 0.0

        predictions_list = []

        labels_list = []

        with torch.no_grad():

            progress_bar = tqdm(
                self.validation_loader,
                desc=f"Validation {epoch+1}/{self.epochs}",
                leave=False
            )

            for batch in progress_bar:

                images = batch["image"].to(self.device)

                labels = batch["label"].to(self.device)

                with autocast():

                    outputs = self.model(images)

                    loss = self.criterion(
                        outputs,
                        labels
                    )

                running_loss += loss.item()

                _, predictions = torch.max(
                    outputs,
                    dim=1
                )

                predictions_list.extend(
                    predictions.cpu().numpy()
                )

                labels_list.extend(
                    labels.cpu().numpy()
                )

        validation_loss = (

            running_loss /

            len(self.validation_loader)

        )

        metrics = Metrics.calculate_metrics(

            labels_list,

            predictions_list

        )

        validation_accuracy = metrics["accuracy"]

        logger.info(

            f"Validation Loss : {validation_loss:.4f}"

        )

        logger.info(

            f"Validation Accuracy : "

            f"{validation_accuracy:.4f}"

        )

        return (

            validation_loss,

            validation_accuracy,

            metrics

        )


    def train(self):
        """
        Complete training pipeline.
        """

        logger.info(
            "Training Started..."
        )

        start_time = time.time()

        for epoch in range(self.epochs):

            train_loss, train_accuracy = (

                self.train_one_epoch(epoch)

            )

            (

                validation_loss,

                validation_accuracy,

                validation_metrics

            ) = self.validate_one_epoch(epoch)

            self.history["train_loss"].append(
                train_loss
            )

            self.history[
                "validation_loss"
            ].append(
                validation_loss
            )

            self.history[
                "train_accuracy"
            ].append(
                train_accuracy
            )

            self.history[
                "validation_accuracy"
            ].append(
                validation_accuracy
            )

            if self.scheduler is not None:

                if (

                    self.scheduler.__class__.__name__

                    == "ReduceLROnPlateau"

                ):

                    self.scheduler.step(

                        validation_loss

                    )

                else:

                    self.scheduler.step()

            is_best = False

            if (

                validation_accuracy

                > self.best_accuracy

            ):

                self.best_accuracy = (

                    validation_accuracy

                )

                self.best_loss = (

                    validation_loss

                )

                self.early_stop_counter = 0

                is_best = True

            else:

                self.early_stop_counter += 1

            self.checkpoint_manager.save_checkpoint(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                epoch=epoch,

                train_loss=train_loss,

                val_loss=validation_loss,

                val_accuracy=validation_accuracy,

                best=is_best

            )

            logger.info(

                "=" * 60

            )

            logger.info(

                f"Epoch : {epoch+1}/{self.epochs}"

            )

            logger.info(

                f"Train Loss : {train_loss:.4f}"

            )

            logger.info(

                f"Validation Loss : "

                f"{validation_loss:.4f}"

            )

            logger.info(

                f"Train Accuracy : "

                f"{train_accuracy:.4f}"

            )

            logger.info(

                f"Validation Accuracy : "

                f"{validation_accuracy:.4f}"

            )

            logger.info(

                f"Precision : "

                f"{validation_metrics['precision']:.4f}"

            )

            logger.info(

                f"Recall : "

                f"{validation_metrics['recall']:.4f}"

            )

            logger.info(

                f"F1 Score : "

                f"{validation_metrics['f1_score']:.4f}"

            )

            logger.info(

                "=" * 60

            )

            patience = self.config.get(

                "training",

                "early_stopping_patience",

                default=5

            )

            if (

                self.early_stop_counter

                >= patience

            ):

                logger.info(

                    "Early Stopping Triggered."

                )

                break

        total_time = (

            time.time()

            - start_time

        )

        logger.info(

            f"Training Completed "

            f"in {total_time/60:.2f} Minutes."

        )

        return self.history


    def load_best_model(self):
        """
        Load the best saved model.
        """

        try:

            checkpoint = self.checkpoint_manager.load_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                best=True
            )

            if checkpoint is not None:

                logger.info(
                    f"Best model loaded "
                    f"(Epoch {checkpoint['epoch'] + 1})"
                )

            return checkpoint

        except Exception as e:

            raise MedicalDiagnosisException(e)

    def get_history(self):
        """
        Returns complete training history.
        """

        return self.history

    def print_summary(self):

        logger.info("=" * 70)

        logger.info("TRAINING SUMMARY")

        logger.info("=" * 70)

        logger.info(
            f"Best Validation Accuracy : "
            f"{self.best_accuracy:.4f}"
        )

        logger.info(
            f"Best Validation Loss : "
            f"{self.best_loss:.4f}"
        )

        logger.info(
            f"Training Epochs : "
            f"{len(self.history['train_loss'])}"
        )

        logger.info("=" * 70)
    