import itertools
from pathlib import Path
from typing import Optional

from revelio.dataset.element import DatasetElementDescriptor, ElementClass

from .loader import DatasetLoader


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
        all_images = sorted(path.rglob("*.png"))
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
        morphed_images = sorted(path.rglob("morph*.png"))
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
        all_images = sorted(path.rglob("*.png"))
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
        morphed_images = sorted(path.rglob("morph*.png"))
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
        all_images = sorted(path.rglob(f"morph_{self._algorithm}/*.jpg"))
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
            for image in all_images
        ]


class FRGCLoader(DatasetLoader):  # pragma: no cover
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = sorted(path.rglob("*.jpg"))
        return [
            DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
            for image in all_images
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
            self._poses = allowed_poses
        elif not set(poses).issubset(allowed_poses):
            raise ValueError(f"Invalid poses: {poses}")
        else:
            self._poses = set(poses)

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = sorted(path.rglob("*.ppm"))
        valid_images = []
        for pose in self._poses:
            pose_images = [img for img in all_images if f"_{pose}.ppm" in img.name]
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
            files = sorted((path / dir_name).rglob("*.jpg"))
            bona_fide.extend(
                DatasetElementDescriptor(x=(image,), y=ElementClass.BONA_FIDE)
                for image in files
            )
        if self._load_morphs:
            dir_name = "londondb_morph_combined_alpha0.5_passport-scale_15kb"
            files = sorted((path / dir_name).rglob("*.jpg"))
            morphed.extend(
                DatasetElementDescriptor(x=(image,), y=ElementClass.MORPHED)
                for image in files
            )
        return bona_fide + morphed
