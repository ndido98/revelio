from pathlib import Path
from typing import Iterable


def rglob_multiple(root: Path, pattern: str, extensions: Iterable[str]) -> list[Path]:
    """
    Find all files matching a pattern with one of the given extensions
    in a directory tree.

    Args:
        root: The root directory to search in.
        pattern: The pattern to match.
        extensions: The extensions to match.
    """
    result = []
    for extension in extensions:
        result.extend(list(root.rglob(f"{pattern}.{extension}")))
    return sorted(result)


def glob_multiple(root: Path, pattern: str, extensions: Iterable[str]) -> list[Path]:
    """
    Find all files matching a pattern with one of the given extensions
    in a directory.

    Args:
        root: The root directory to search in.
        pattern: The pattern to match.
        extensions: The extensions to match.
    """
    result = []
    for extension in extensions:
        result.extend(list(root.glob(f"{pattern}.{extension}")))
    return sorted(result)
