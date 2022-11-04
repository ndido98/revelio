import argparse
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from pydantic import ValidationError
from torch.utils.data import DataLoader
from tqdm import tqdm

from revelio.config import Config
from revelio.dataset import Dataset, DatasetFactory
from revelio.model import Model
from revelio.utils.iterators import consume
from revelio.utils.logging import TqdmLoggingHandler
from revelio.utils.random import set_seed


def _is_valid_device(
    device: str, is_cuda_available: bool, cuda_device_count: int
) -> str:
    if device == "cpu":
        return device
    elif device.startswith("cuda"):
        if not is_cuda_available:
            raise argparse.ArgumentTypeError("CUDA is not available on this computer")
        # We should have either an empty string or a colon, followed by a number
        device_rest = device[4:]
        if len(device_rest) == 0:
            return device
        if len(device_rest) > 1 and device_rest[0] == ":":
            try:
                device_index = int(device_rest[1:])
            except ValueError as e:
                raise argparse.ArgumentTypeError(
                    "CUDA device index must be an integer"
                ) from e
            device_count = cuda_device_count
            if device_index >= device_count:
                if device_count == 1:
                    raise argparse.ArgumentTypeError(
                        f"Invalid cuda device index: got {device_index}, expected 0"
                    )
                else:
                    raise argparse.ArgumentTypeError(
                        f"Invalid CUDA device index: got {device_index}, "
                        f"expected between 0 and {device_count - 1} (inclusive)"
                    )
            return device
        else:
            raise argparse.ArgumentTypeError(f"Invalid device: {device}")
    else:
        raise argparse.ArgumentTypeError(f"Invalid device: {device}")


def _valid_device(device: str) -> str:
    try:
        return _is_valid_device(
            device, torch.cuda.is_available(), torch.cuda.device_count()
        )
    except argparse.ArgumentTypeError:
        raise


def _verbose_count_to_logging_level(level: int) -> int:
    match level:
        case 0:
            return logging.WARNING
        case 1:
            return logging.INFO
        case _:
            return logging.DEBUG


def _create_warmup_data_loader(dataset: Dataset, workers_count: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers_count,
    )


def _create_data_loader(
    dataset: Dataset, batch_size: int, workers_count: int, persistent_workers: bool
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers_count,
        persistent_workers=persistent_workers if workers_count > 0 else False,
        pin_memory=True,
    )


def _warmup(
    config: Config,
    warmup_workers_count: int,
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
) -> None:
    if config.experiment.training.enabled:
        train_ds.warmup = True
        val_ds.warmup = True
        warmup_train_dl = _create_warmup_data_loader(train_ds, warmup_workers_count)
        warmup_val_dl = _create_warmup_data_loader(val_ds, warmup_workers_count)
    test_ds.warmup = True
    warmup_test_dl = _create_warmup_data_loader(test_ds, warmup_workers_count)
    # Warmup (i.e. run the offline processing) the three data loaders so we don't
    # have an overhead when we start training
    if config.experiment.training.enabled:
        if config.seed is not None:
            set_seed(config.seed)
        consume(tqdm(warmup_train_dl, desc="Warming up the training data loader"))
        if config.seed is not None:
            set_seed(config.seed)
        consume(tqdm(warmup_val_dl, desc="Warming up the validation data loader"))
    if config.seed is not None:
        set_seed(config.seed)
    consume(tqdm(warmup_test_dl, desc="Warming up the test data loader"))
    train_ds.warmup = False
    val_ds.warmup = False
    test_ds.warmup = False


def _get_report_dict(
    experiment_name: str,
    experiment_start: datetime,
    experiment_end: datetime,
    metrics: Mapping[str, Mapping[str, npt.ArrayLike]],
) -> dict[str, Any]:
    def _dataset_metrics_to_dict(
        ds_metrics: Mapping[str, npt.ArrayLike]
    ) -> dict[str, Any]:
        res = {}
        for k, v in ds_metrics.items():
            if isinstance(v, np.ndarray):
                res[k] = v.tolist()
            else:
                res[k] = v
        return res

    return {
        "meta": {
            "name": experiment_name,
            "start_time": experiment_start.isoformat(),
            "end_time": experiment_end.isoformat(),
        },
        "metrics": {ds: _dataset_metrics_to_dict(m) for ds, m in metrics.items()},
    }


def _cli_program(args: Any) -> None:
    config = Config.from_string(args.config_file.read())
    experiment_name = Path(args.config_file.name).stem
    experiment_start = datetime.now()
    # Set the seed as soon as possible
    if config.seed is not None:
        set_seed(config.seed)
    else:
        seed = random.randint(0, 2**32 - 1)
        print(f"No seed was specified, using a random seed: {seed}")
        set_seed(seed)
    print("Loading the dataset...")
    dataset = DatasetFactory(config)
    train_ds = dataset.get_train_dataset()
    val_ds = dataset.get_val_dataset()
    test_ds = dataset.get_test_dataset()
    if args.warmup:
        _warmup(config, args.warmup_workers_count, train_ds, val_ds, test_ds)
    train_dl = _create_data_loader(
        train_ds,
        batch_size=config.experiment.batch_size,
        workers_count=args.workers_count,
        persistent_workers=args.persistent_workers,
    )
    val_dl = _create_data_loader(
        val_ds,
        batch_size=config.experiment.batch_size,
        workers_count=args.workers_count,
        persistent_workers=args.persistent_workers,
    )
    test_dl = _create_data_loader(
        test_ds,
        batch_size=config.experiment.batch_size,
        workers_count=args.workers_count,
        persistent_workers=args.persistent_workers,
    )
    # Call the data loaders iterators to create now the persistent workers
    if args.persistent_workers and config.experiment.training.enabled:
        print("Creating the workers...")
        iter(train_dl)
        iter(val_dl)
    model = Model.find(
        config.experiment.model.name,
        config=config,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        test_dataloader=test_dl,
        device=args.device,
    )
    if config.experiment.training.enabled:
        print("Fitting the model...")
        model.fit()
    del train_dl, train_ds
    del val_dl, val_ds
    print("Evaluating the model...")
    metrics = model.evaluate()
    experiment_end = datetime.now()
    if config.experiment.scores.metrics is not None:
        report_dict = _get_report_dict(
            experiment_name, experiment_start, experiment_end, metrics
        )
        report_json = json.dumps(report_dict, indent=4)
        config.experiment.scores.metrics.write_text(report_json)
    print("---- EXPERIMENT RESULTS ----")
    print(f"Experiment name: {experiment_name}")
    print(f"Experiment start: {experiment_start}")
    print(f"Experiment end: {experiment_end}")
    for ds, metric in metrics.items():
        print(f"Metrics for dataset {ds}:")
        for k, v in metric.items():
            print(f"\t{k}: {v!r}")
    print("Scores have been saved in the following files:")
    print(f"\tBona fide: {config.experiment.scores.bona_fide}")
    print(f"\tMorphed: {config.experiment.scores.morphed}")
    if config.experiment.scores.metrics is not None:
        print(f"\tMetrics: {config.experiment.scores.metrics}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revelio - A declarative framework for face morphing detection experiments"
        )
    )
    parser.add_argument(
        "config_file",
        metavar="CONFIG_FILE",
        type=argparse.FileType("r", encoding="utf-8"),
        help="The configuration file of the experiment",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=_valid_device,
        default="cpu",
        help="The device to use for the experiment",
    )
    parser.add_argument(
        "-w",
        "--workers-count",
        type=int,
        default=0,
        help="The number of workers that the data loader should use",
    )
    parser.add_argument(
        "-W",
        "--warmup-workers-count",
        type=int,
        default=0,
        help="The number of workers that the data loader should use for the warmup",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Set this to true to enable verbose output",
        dest="logging_level",
    )
    parser.add_argument(
        "--no-persistent-workers",
        action="store_const",
        const=False,
        default=True,
        help="Set this to true to disable persistent workers during training",
        dest="persistent_workers",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_const",
        const=False,
        default=True,
        help="Set this to true to disable the warmup",
        dest="warmup",
    )

    args = parser.parse_args()

    logging_level = _verbose_count_to_logging_level(args.logging_level)
    logging.basicConfig(
        handlers=[TqdmLoggingHandler(logging_level)],
        level=logging_level,
        force=True,
    )
    plt.switch_backend("agg")

    try:
        _cli_program(args)
    except (TypeError, ValueError, ValidationError) as e:
        # Ignore pretty printing of exceptions and just re-raise them
        if logging_level > 0:
            raise
        print(
            "---------------------------- FATAL ERROR ----------------------------\n"
            "Revelio encountered a fatal error while trying to run the experiment.\n"
            "Please check the error message below and try to fix the problem.\n"
            "If you think this is a bug in Revelio, please open an issue on GitHub:\n"
            "https://github.com/ndido98/revelio/issues\n"
            "----------------------------------------------------------------------\n\n"
            f"{e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
