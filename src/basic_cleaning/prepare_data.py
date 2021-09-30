#!/usr/bin/env python
"""
The basic_cleaning step is a component in this MLflow pipeline using Weights and Biases.
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    ######################
    # YOUR CODE HERE     #
    ######################
    local_path = wandb.use_artifact("sample.csv:latest").file()
    df = pd.read_csv(local_path)
    logger.info("INFO: Read input data: {0}, number of records: {1}".format(local_path, df.shape[0]))
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info("INFO: Removed records with outliers in dataframe, number of records: {0}".format(df.shape[0]))

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    logger.info("SUCCESS: Saved and uploaded modified input file to wandb")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This conponent prepares the input data for downstream analyses")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Provide filename path for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Provide filename path for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Provide type for the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Provide description for the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Provide minimum price threshold for pruning data",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Provide maximum price threshold for pruning data",
        required=True
    )

    args = parser.parse_args()

    go(args)
