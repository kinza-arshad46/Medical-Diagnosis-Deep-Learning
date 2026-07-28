"""
==========================================================
Medical Diagnosis AI
Configuration Loader
==========================================================
Author : Kinza Arshad
Description:
Loads and validates the YAML configuration file.
==========================================================
"""

from pathlib import Path
import yaml


class ConfigLoader:
    """
    A utility class to load project configuration from YAML.
    """

    def __init__(self, config_path="config/config.yaml"):
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        self.config = self._load_yaml()

    def _load_yaml(self):
        """
        Load YAML configuration file.
        """

        try:
            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            return config

        except yaml.YAMLError as e:
            raise ValueError(
                f"Error while reading YAML configuration:\n{e}"
            )

    def get(self, *keys, default=None):
        """
        Get nested configuration values.

        Example:
            config.get("training", "batch_size")
        """

        value = self.config

        try:
            for key in keys:
                value = value[key]

            return value

        except (KeyError, TypeError):
            return default

    def show(self):
        """
        Print complete configuration.
        """

        from pprint import pprint

        pprint(self.config)


if __name__ == "__main__":

    config = ConfigLoader()

    print("Project Name:", config.get("project", "name"))

    print("Model:", config.get("model", "name"))

    print("Epochs:", config.get("training", "epochs"))

    print("Image Size:", config.get("image", "image_size"))