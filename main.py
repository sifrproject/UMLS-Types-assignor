import os
import sys
import argparse

# Typing
from typing import Any

# Config File
import yaml

# Pipeline
import mlflow

from get_data import generate_source_data
from process_data import preprocess
from train import train_and_test

CONFIG_PATH = "./"

# Function to load yaml configuration file


def load_config(config_name: str) -> Any:
    """Load config file

    Args:
        config_name (str): Path of the config file

    Returns:
        Any: Config dict
    """
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        return yaml.safe_load(file)


def main():
    """Main function of the pipeline
    """
    config = load_config("config.yaml")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--verbose', help='Active verbose mode.', action='store_true')
    parser.add_argument('--only_source', help='Pipeline launchs only the generation of the \
        source data.')
    parser.add_argument('--only_preprocess', help='Pipeline launchs only the preprocess of the \
        source data.')
    parser.add_argument('--only_training', help='Pipeline launchs only the training of the \
        preprocessed data.')
    parser.add_argument('--limit', type=int,
                        help='Limit of the source data number generated.')
    parser.add_argument('--debug_output_path', type=str,
                        help='Path of the output log.')

    args = parser.parse_args()

    if args.verbose:
        config["verbose"] = True
    else:
        config["verbose"] = False

    if args.debug_output_path:
        config["debug_output_path"] = args.debug_output_path

    only_source = args.only_source
    only_preprocess = args.only_preprocess
    only_training = args.only_training
    limit = args.limit if args.limit else None

    if limit is None and config["verbose"]:
        answer = input("You are running pipeline with the entire source data. \
            It'll be long to generate all data. Are you sure to continue ? [y/n]")
        if answer == 'n':
            print("Quitting")
            sys.exit(0)

    all = not (only_source or only_preprocess or only_training)

    with mlflow.start_run(run_name="test_mlflow"):

        # Print out current run_uuid
        run_uuid = mlflow.active_run().info.run_uuid
        print(f"MLflow Run ID: {str(run_uuid)}")
        mlflow.log_param("run_uuid", run_uuid)

        if only_source or all:
            generate_source_data(limit, config["verbose"])
        if only_preprocess or all:
            preprocess(config)
        if only_training or all:
            train_and_test(config)


if __name__ == "__main__":
    main()
