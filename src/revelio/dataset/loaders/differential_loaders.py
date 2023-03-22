from pathlib import Path
from typing import Literal, Optional

from revelio.dataset.element import DatasetElementDescriptor, ElementClass
from revelio.utils.files import rglob_multiple

from .loader import DatasetLoader

IMAGE_EXTENSIONS = ("png", "jpg", "jpeg", "tiff", "tif", "jp2", "ppm")

MorphCoupleType = Literal["none", "accomplice", "criminal", "both"]


def _parse_file_names(value: Optional[str | list[str]], default: str) -> list[str]:
    if value is None:
        return [default]
    if isinstance(value, str):
        return [value]
    return value


class DifferentialPMDBLoader(DatasetLoader):
    def __init__(
        self,
        morph_couple_type: MorphCoupleType = "both",
        bona_fide_file_names: Optional[str | list[str]] = None,
        criminal_file_names: Optional[str | list[str]] = None,
        accomplice_file_names: Optional[str | list[str]] = None,
    ) -> None:
        self._morph_couple_type = morph_couple_type
        self._bona_fide_file_names = _parse_file_names(
            bona_fide_file_names, "couples_pmdb_bona_fine.txt"
        )
        self._criminal_file_names = _parse_file_names(
            criminal_file_names, "couples_pmdb_morphed_criminal_0.55.txt"
        )
        self._accomplice_file_names = _parse_file_names(
            accomplice_file_names, "couples_pmdb_morphed_accomplice_0.55.txt"
        )

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        couples = []
        for file_name in self._bona_fide_file_names:
            couples += self._load_couples_file(
                path / file_name, path, ElementClass.BONA_FIDE
            )
        if self._morph_couple_type in ("criminal", "both"):
            for file_name in self._criminal_file_names:
                couples += self._load_couples_file(
                    path / file_name, path, ElementClass.MORPHED_WITH_CRIMINAL
                )
        if self._morph_couple_type in ("accomplice", "both"):
            for file_name in self._accomplice_file_names:
                couples += self._load_couples_file(
                    path / file_name, path, ElementClass.MORPHED_WITH_ACCOMPLICE
                )
        return couples

    def _load_couples_file(
        self, couples_file: Path, root: Path, element_class: ElementClass
    ) -> list[DatasetElementDescriptor]:
        couples = []
        lines = couples_file.read_text().splitlines()
        for line in lines:
            probe_name, live_name = line.split()
            probe, live = root / probe_name, root / live_name
            couples.append(DatasetElementDescriptor(x=(probe, live), y=element_class))
        return couples


class DifferentialIdiapLoader(DatasetLoader):
    def __init__(
        self,
        bona_fide_root: str,
        morphed_root: Optional[str] = None,
        morph_couple_type: MorphCoupleType = "both",
        bona_fide_file_name: str = "BonafideAttemptIndex.txt",
        criminal_file_name: str = "Criminal_MorphAttemptIndex.txt",
        accomplice_file_name: str = "Accomplice_MorphAttemptIndex.txt",
        load_bona_fide_couples: bool = True,
    ) -> None:
        self._bona_fide_root = Path(bona_fide_root)
        self._morphed_root = None if morphed_root is None else Path(morphed_root)
        self._morph_couple_type = morph_couple_type
        self._bona_fide_file_name = bona_fide_file_name
        self._criminal_file_name = criminal_file_name
        self._accomplice_file_name = accomplice_file_name
        self._load_bona_fide_couples = load_bona_fide_couples

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        couples = []
        if self._load_bona_fide_couples:
            bona_fide_couples = path / self._bona_fide_file_name
            couples += self._load_couples_file(
                bona_fide_couples, self._bona_fide_root, ElementClass.BONA_FIDE
            )
        if self._morph_couple_type in ("criminal", "both"):
            if self._morphed_root is None:
                raise ValueError("Morphed root must be specified")
            criminal_morph_couples = path / self._criminal_file_name
            couples += self._load_couples_file(
                criminal_morph_couples,
                self._morphed_root,
                ElementClass.MORPHED_WITH_CRIMINAL,
            )
        if self._morph_couple_type in ("accomplice", "both"):
            if self._morphed_root is None:
                raise ValueError("Morphed root must be specified")
            accomplice_morph_couples = path / self._accomplice_file_name
            couples += self._load_couples_file(
                accomplice_morph_couples,
                self._morphed_root,
                ElementClass.MORPHED_WITH_ACCOMPLICE,
            )
        return couples

    def _load_couples_file(
        self, couples_file: Path, probe_root: Path, element_class: ElementClass
    ) -> list[DatasetElementDescriptor]:
        def _get_files_index(path: Path) -> dict[str, list[Path]]:
            images = rglob_multiple(path, "*", IMAGE_EXTENSIONS)
            files_index: dict[str, list[Path]] = {}
            for img in images:
                stem = img.stem
                if stem not in files_index:
                    files_index[stem] = []
                files_index[stem].append(img)
            return files_index

        couples = []
        lines = couples_file.read_text().splitlines()
        bona_fide_images = _get_files_index(self._bona_fide_root)
        probe_images = _get_files_index(probe_root)
        for line in lines:
            probe_name, live_name = line.split()
            probe, live = Path(probe_name), Path(live_name)
            # Find an image with the given stem in the probe root
            probe_matches = probe_images.get(probe.stem, [])
            if len(probe_matches) == 0:
                print(list(rglob_multiple(probe_root, "*", IMAGE_EXTENSIONS))[:100])
                raise ValueError(
                    f"Could not find a probe image with name {probe} in {probe_root}"
                )
            elif len(probe_matches) > 1:
                raise ValueError(
                    f"Found multiple probe images with name {probe} in {probe_root}"
                )
            probe_found = probe_matches[0]
            # Find the live image with the given stem in the bona fide root
            live_matches = bona_fide_images.get(live.stem, [])
            if len(live_matches) == 0:
                raise ValueError(
                    f"Could not find a live image with name {live} "
                    f"in {self._bona_fide_root}"
                )
            elif len(live_matches) > 1:
                raise ValueError(
                    f"Found multiple live images with name {live} "
                    f"in {self._bona_fide_root}"
                )
            live_found = live_matches[0]
            couples.append(
                DatasetElementDescriptor(x=(probe_found, live_found), y=element_class)
            )
        return couples


class DifferentialMorphDBLoader(DatasetLoader):
    def __init__(
        self,
        morph_couple_type: MorphCoupleType = "both",
        include_print_scan: bool = False,
        bona_fide_file_name: str = "couples_morphdb_bona_fide_ALL.txt",
        criminal_file_name: str = "couples_morphdb_morphed_criminal.txt",
        accomplice_file_name: str = "couples_morphdb_morphed_accomplice.txt",
    ) -> None:
        self._morph_couple_type = morph_couple_type
        self._include_print_scan = include_print_scan
        self._bona_fide_file_name = bona_fide_file_name
        self._criminal_file_name = criminal_file_name
        self._accomplice_file_name = accomplice_file_name

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        bona_fide_couples = path / self._bona_fide_file_name
        criminal_morph_couples = path / self._criminal_file_name
        accomplice_morph_couples = path / self._accomplice_file_name
        couples = self._load_couples_file(
            bona_fide_couples, path, ElementClass.BONA_FIDE
        )
        if self._morph_couple_type in ("criminal", "both"):
            couples += self._load_couples_file(
                criminal_morph_couples, path, ElementClass.MORPHED_WITH_CRIMINAL
            )
        if self._morph_couple_type in ("accomplice", "both"):
            couples += self._load_couples_file(
                accomplice_morph_couples, path, ElementClass.MORPHED_WITH_ACCOMPLICE
            )
        return couples

    def _load_couples_file(
        self, couples_file: Path, root: Path, element_class: ElementClass
    ) -> list[DatasetElementDescriptor]:
        couples = []
        lines = couples_file.read_text().splitlines()
        for line in lines:
            probe_name, live_name = line.split()
            probe, live = root / probe_name, root / live_name
            couples.append(DatasetElementDescriptor(x=(probe, live), y=element_class))
        return couples


class DifferentialFEILoader(DatasetLoader):
    def __init__(
        self,
        bona_fide_root: str,
        morphed_root: Optional[str] = None,
        morph_couple_type: MorphCoupleType = "both",
        bona_fide_file_name: str = "bonafide.txt",
        criminal_file_name: str = "criminal.txt",
        accomplice_file_name: str = "accomplice.txt",
        load_bona_fide_couples: bool = True,
    ) -> None:
        self._bona_fide_root = Path(bona_fide_root)
        self._morphed_root = None if morphed_root is None else Path(morphed_root)
        self._morph_couple_type = morph_couple_type
        self._bona_fide_file_name = bona_fide_file_name
        self._criminal_file_name = criminal_file_name
        self._accomplice_file_name = accomplice_file_name
        self._load_bona_fide_couples = load_bona_fide_couples

    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        couples = []
        if self._load_bona_fide_couples:
            bona_fide_couples = path / self._bona_fide_file_name
            couples += self._load_couples_file(
                bona_fide_couples,
                self._bona_fide_root,
                self._bona_fide_root,
                ElementClass.BONA_FIDE,
            )
        if self._morph_couple_type in ("criminal", "both"):
            if self._morphed_root is None:
                raise ValueError("Morphed root must be specified")
            criminal_morph_couples = path / self._criminal_file_name
            couples += self._load_couples_file(
                criminal_morph_couples,
                self._morphed_root,
                self._bona_fide_root,
                ElementClass.MORPHED_WITH_CRIMINAL,
            )
        if self._morph_couple_type in ("accomplice", "both"):
            if self._morphed_root is None:
                raise ValueError("Morphed root must be specified")
            accomplice_morph_couples = path / self._accomplice_file_name
            couples += self._load_couples_file(
                accomplice_morph_couples,
                self._morphed_root,
                self._bona_fide_root,
                ElementClass.MORPHED_WITH_ACCOMPLICE,
            )
        return couples

    def _load_couples_file(
        self,
        couples_file: Path,
        probe_root: Path,
        live_root: Path,
        element_class: ElementClass,
    ) -> list[DatasetElementDescriptor]:
        couples = []
        lines = couples_file.read_text().splitlines()
        for line in lines:
            probe_name, live_name = line.split()
            probe, live = probe_root / probe_name, live_root / live_name
            couples.append(DatasetElementDescriptor(x=(probe, live), y=element_class))
        return couples
