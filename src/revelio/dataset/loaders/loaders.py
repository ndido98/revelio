import itertools
from pathlib import Path
from typing import Optional

from revelio.dataset.element import DatasetElementDescriptor, ElementClass
from revelio.utils.files import rglob_multiple

from .loader import DatasetLoader

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "tiff", "tif", "jp2", "ppm")


class PMDBLoader(DatasetLoader):  # pragma: no cover
    def __init__(
        self,
        include_training_bona_fide: bool = False,
        morph_percentages_count: int = 1,
    ):
        if morph_percentages_count < 0:
            raise ValueError("morph_percentages_count must be non-negative")
        self._include_train_bf = include_training_bona_fide
        self._morph_percentages_count = morph_percentages_count

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        bona_fide = []
        morphed = []
        for image in all_images:
            if "morph" not in image.name and (
                self._include_train_bf or image.parent.name == "TestImages"
            ):
                bona_fide.append(
                    DatasetElementDescriptor(
                        x=(image,),
                        y=ElementClass.BONA_FIDE,
                    )
                )
        morphed_images = rglob_multiple(path, "morph*", IMAGE_EXTENSIONS)
        # Take at most n morphed images
        for _, g in itertools.groupby(morphed_images, lambda x: x.stem[:-4]):
            group_images = list(g)
            morphed_count = min(self._morph_percentages_count, len(group_images))
            for i in range(morphed_count):
                morphed.append(
                    DatasetElementDescriptor(
                        x=(group_images[i],),
                        y=ElementClass.MORPHED,
                    )
                )
        return bona_fide + morphed


class MorphDBLoader(DatasetLoader):  # pragma: no cover
    def __init__(
        self,
        kinds: list[str],
        include_training_bona_fide: bool = False,
    ):
        allowed_kinds = {"digital", "ps300dpi", "ps600dpi"}
        if not set(kinds).issubset(allowed_kinds):
            raise ValueError(f"Invalid kinds: {kinds}")
        self._kinds = set(kinds)
        self._include_train_bf = include_training_bona_fide

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        bona_fide = []
        morphed = []
        for image in all_images:
            if (
                ("morph" not in image.name)
                and (self._include_train_bf or image.parent.name == "TestImages")
                and (
                    (
                        "digital" in self._kinds
                        and "_D" not in image.stem
                        and "_PS" not in image.stem
                    )
                    or ("ps300dpi" in self._kinds and "_PS300" in image.stem)
                    or ("ps600dpi" in self._kinds and "_PS600" in image.stem)
                )
            ):
                bona_fide.append(
                    DatasetElementDescriptor(
                        x=(image,),
                        y=ElementClass.BONA_FIDE,
                    )
                )
        morphed_images = rglob_multiple(path, "morph*", IMAGE_EXTENSIONS)
        for image in morphed_images:
            if (
                (
                    "digital" in self._kinds
                    and "_D" not in image.stem
                    and "_PS" not in image.stem
                )
                or ("ps300dpi" in self._kinds and "_PS300dpiR" in image.stem)
                or ("ps600dpi" in self._kinds and "_PS600dpiR" in image.stem)
            ):
                morphed.append(
                    DatasetElementDescriptor(
                        x=(image,),
                        y=ElementClass.MORPHED,
                    )
                )

        return bona_fide + morphed


class IdiapMorphedLoader(DatasetLoader):  # pragma: no cover
    def __init__(self, algorithm: str) -> None:
        self._algorithm = algorithm

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        images = rglob_multiple(path, f"morph_{self._algorithm}/*", IMAGE_EXTENSIONS)
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
            for image in images
        ]


class FRGCLoader(DatasetLoader):  # pragma: no cover
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
            for image in images
        ]


class FERETLoader(DatasetLoader):  # pragma: no cover
    def __init__(self, poses: Optional[list[str]] = None):
        # fmt: off
        allowed_poses = {
            "fa", "fb",        # Frontal
            "pl", "hl", "ql",  # Left (profile, half, quarter)
            "pr", "hr", "qr",  # Right (profile, half, quarter)
            "ra", "rb",        # Left-rotated random image
            "rc", "rd", "re",  # Right-rotated random image
        }
        # fmt: on
        if poses is None:
            self._poses = {"fa", "fb"}  # Load only frontal images as default
        elif not set(poses).issubset(allowed_poses):
            raise ValueError(f"Invalid poses: {poses}")
        else:
            self._poses = set(poses)

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        valid_images = []
        for pose in self._poses:
            pose_images = [img for img in all_images if f"_{pose}." in img.name]
            pose_alt_images = [img for img in all_images if f"_{pose}_" in img.stem]
            valid_images.extend(pose_images)
            valid_images.extend(pose_alt_images)
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
            for image in valid_images
        ]


class AMSLLoader(DatasetLoader):  # pragma: no cover
    def __init__(self, poses: Optional[list[str]] = None, load_morphs: bool = True):
        allowed_poses = {"neutral", "smiling"}
        if poses is None:
            self._poses = allowed_poses
        elif not set(poses).issubset(allowed_poses):
            raise ValueError(f"Invalid poses: {poses}")
        else:
            self._poses = set(poses)
        self._load_morphs = load_morphs

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        bona_fide: list[DatasetElementDescriptor] = []
        morphed: list[DatasetElementDescriptor] = []
        for pose in self._poses:
            dir_name = f"londondb_genuine_{pose}_passport-scale_15kb"
            files = rglob_multiple(path / dir_name, "*", IMAGE_EXTENSIONS)
            bona_fide.extend(
                DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
                for image in files
            )
        if self._load_morphs:
            dir_name = "londondb_morph_combined_alpha0.5_passport-scale_15kb"
            files = rglob_multiple(path / dir_name, "*", IMAGE_EXTENSIONS)
            morphed.extend(
                DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
                for image in files
            )
        return bona_fide + morphed


class BiometixMorphedLoader(DatasetLoader):  # pragma: no cover
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
            for image in images
        ]


class CFDLoader(DatasetLoader):  # pragma: no cover
    def __init__(self, poses: Optional[list[str]] = None):
        allowed_poses = {"n", "a", "f", "hc", "ho"}
        if poses is None:
            self._poses = {"n"}  # Load only the neutral pose as default
        else:
            lower_poses = {p.lower() for p in poses}
            if not lower_poses.issubset(allowed_poses):
                raise ValueError(f"Invalid poses: {poses}")
            else:
                self._poses = lower_poses

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        valid_images = []
        for pose in self._poses:
            pose_images = [
                img
                for img in images
                if img.stem.lower().split("-")[-1].startswith(pose)
            ]
            valid_images.extend(pose_images)
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
            for image in valid_images
        ]


class CFDMorphLoader(DatasetLoader):  # pragma: no cover
    def __init__(
        self,
        algorithms: Optional[list[str]] = None,
        morph_levels: Optional[list[float]] = None,
    ):
        allowed_algorithms = {"c02", "c03", "c05", "c08"}
        if algorithms is None:
            self._algorithms = allowed_algorithms
        else:
            lower_algorithms = {a.lower() for a in algorithms}
            if not lower_algorithms.issubset(allowed_algorithms):
                raise ValueError(f"Invalid algorithms: {algorithms}")
            else:
                self._algorithms = lower_algorithms
        self._morph_levels = (
            [int(level * 100) for level in morph_levels]
            if morph_levels is not None
            else None
        )

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
        valid_images: list[Path] = []
        for algorithm in self._algorithms:
            if self._morph_levels is None:
                valid_images.extend(
                    img for img in images if f"_{algorithm}_" in img.stem.lower()
                )
            else:
                for level in self._morph_levels:
                    valid_images.extend(
                        img
                        for img in images
                        if f"_{algorithm}_b{level}_" in img.stem.lower()
                    )
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
            for image in valid_images
        ]
