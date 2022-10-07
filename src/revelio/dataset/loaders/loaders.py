import itertools
from pathlib import Path

from revelio.dataset.element import DatasetElementDescriptor, ElementClass

from .loader import DatasetLoader


class PMDBLoader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = sorted(path.rglob("*.png"))
        bona_fide = []
        morphed = []
        for image in all_images:
            if "morph" not in image.name and "keypoints" not in image.name:
                bona_fide.append(
                    DatasetElementDescriptor(
                        x=(image,),
                        y=ElementClass.BONA_FIDE,
                    )
                )
        morphed_images = sorted(path.rglob("morph*.png"))
        # Take at most 4 morphed images
        for _, g in itertools.groupby(morphed_images, lambda x: x.stem[:-4]):
            group_images = list(g)
            morphed_count = min(4, len(group_images))
            for i in range(morphed_count):
                morphed.append(
                    DatasetElementDescriptor(
                        x=(group_images[i],),
                        y=ElementClass.MORPHED,
                    )
                )
        return bona_fide + morphed


class MorphDBLoader(DatasetLoader):
    def load(self, path: Path) -> list[DatasetElementDescriptor]:
        all_images = sorted(path.rglob("*.png"))
        bona_fide = []
        morphed = []
        for image in all_images:
            if "morph" not in image.name and "keypoints" not in image.name:
                bona_fide.append(
                    DatasetElementDescriptor(
                        x=(image,),
                        y=ElementClass.BONA_FIDE,
                    )
                )
        morphed_images = sorted(path.rglob("morph*.png"))
        # Take at most 4 morphed images
        for _, g in itertools.groupby(morphed_images, lambda x: x.stem[:-4]):
            group_images = list(g)
            morphed_count = min(4, len(group_images))
            for i in range(morphed_count):
                morphed.append(
                    DatasetElementDescriptor(
                        x=(group_images[i],),
                        y=ElementClass.MORPHED,
                    )
                )
        return bona_fide + morphed
