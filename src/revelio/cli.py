import argparse
import random
import sys

import numpy as np
import torch
from pydantic import ValidationError

from revelio.config import Config
from revelio.dataset import DatasetFactory


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


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
        "-v",
        "--verbose",
        action="store_const",
        const=True,
        default=False,
        help="Set this to true to enable verbose output",
    )

    args = parser.parse_args()

    try:
        config = Config.from_string(args.config_file.read())
        # Set the seed as soon as possible
        if config.experiment.seed is not None:
            set_seed(config.experiment.seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            print(f"No seed was specified, using a random seed: {seed}")
            set_seed(seed)
        dataset = DatasetFactory(config)
        print("Dataset:", dataset)
    except (TypeError, ValueError, ValidationError) as e:
        # Ignore pretty printing of exceptions and just re-raise them
        if args.verbose:
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
