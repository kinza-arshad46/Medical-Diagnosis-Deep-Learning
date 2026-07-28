"""
==========================================================
Medical Diagnosis AI
Checkpoint Manager
==========================================================

Author : Kinza Arshad

Description:
Handles saving and loading model checkpoints.
==========================================================
"""


from pathlib import Path
import torch

from src.utils.config_loader import ConfigLoader
from src.utils.logger import logger
from src.utils.custom_exception import MedicalDiagnosisException



class CheckpointManager:


    def __init__(self):

        try:

            self.config = ConfigLoader()


            self.model_directory = Path(
                self.config.get(
                    "paths",
                    "model_dir"
                )
            )


            self.model_directory.mkdir(
                parents=True,
                exist_ok=True
            )


            self.best_model_path = (
                self.model_directory /
                "best_model.pth"
            )


            self.latest_model_path = (
                self.model_directory /
                "latest_checkpoint.pth"
            )


        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Save Checkpoint
    # --------------------------------------------------

    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        epoch,
        train_loss,
        val_loss,
        val_accuracy,
        best=False
    ):


        try:


            checkpoint = {

                "epoch": epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict()
                    if optimizer
                    else None,

                "scheduler_state_dict":
                    scheduler.state_dict()
                    if scheduler
                    else None,

                "train_loss":
                    train_loss,

                "validation_loss":
                    val_loss,

                "validation_accuracy":
                    val_accuracy
            }



            torch.save(
                checkpoint,
                self.latest_model_path
            )


            logger.info(
                "Latest checkpoint saved."
            )



            if best:

                torch.save(
                    checkpoint,
                    self.best_model_path
                )


                logger.info(
                    "Best checkpoint saved."
                )



        except Exception as e:

            raise MedicalDiagnosisException(e)



    # --------------------------------------------------
    # Load Checkpoint
    # --------------------------------------------------

    def load_checkpoint(
        self,
        model,
        optimizer=None,
        scheduler=None,
        best=True
    ):


        try:


            checkpoint_path = (

                self.best_model_path

                if best

                else self.latest_model_path

            )



            if not checkpoint_path.exists():


                logger.warning(
                    f"Checkpoint not found: {checkpoint_path}"
                )


                return None



            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )



            checkpoint = torch.load(
                checkpoint_path,
                map_location=device
            )



            # ---------------------------------------
            # Extract model weights
            # ---------------------------------------

            if (
                "model_state_dict"
                in checkpoint
            ):

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            else:

                state_dict = checkpoint



            # ---------------------------------------
            # Remove DataParallel prefix
            # ---------------------------------------

            new_state_dict = {}

            for key, value in state_dict.items():

                new_key = key.replace(
                    "module.",
                    ""
                )

                new_state_dict[new_key] = value



            model.load_state_dict(
                new_state_dict,
                strict=True
            )



            logger.info(
                "Model checkpoint loaded successfully."
            )



            if optimizer is not None:

                if (
                    checkpoint.get(
                        "optimizer_state_dict"
                    )
                    is not None
                ):

                    optimizer.load_state_dict(
                        checkpoint[
                            "optimizer_state_dict"
                        ]
                    )



            if scheduler is not None:

                if (
                    checkpoint.get(
                        "scheduler_state_dict"
                    )
                    is not None
                ):

                    scheduler.load_state_dict(
                        checkpoint[
                            "scheduler_state_dict"
                        ]
                    )



            return checkpoint



        except Exception as e:

            raise MedicalDiagnosisException(e)
