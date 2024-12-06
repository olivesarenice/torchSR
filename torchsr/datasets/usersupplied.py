import os
from typing import Callable, List, Optional, Union

from .common import FolderByDir, pil_loader


class UserSupplied(FolderByDir):
    """`UserSupplied <https://data.vision.ee.ethz.ch/cvl/UserSupplied/>` Superresolution Dataset

    Args:
        root (string): Root directory for the dataset.
        scale (int, optional): The upsampling ratio: 2, 3, 4 or 8.
        track (str, optional): The downscaling method: bicubic, unknown, real_mild,
            real_difficult, real_wild.
        split (string, optional): The dataset split, supports ``train``, ``val`` or 'test'.
        transform (callable, optional): A function/transform that takes in several PIL images
            and returns a transformed version. It is not a torchvision transform!
        loader (callable, optional): A function to load an image given its path.
        download (boolean, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        predecode (boolean, optional): If true, decompress the image files to disk
        preload (boolean, optional): If true, load all images in memory
    """

    usersupplied_dir = (
        "/home/oliver/ADRA/experiments-superres/dataset/hi_res_focus_zip/bw"
    )

    # # URL paths for the dataset
    # urls = [
    #     # Training datasets
    #     (f"{usersupplied_dir}/UserSupplied_train_HR.zip", None),
    #     (f"{usersupplied_dir}/UserSupplied_train_LR_bicubic_X8.zip", None),
    #     (f"{usersupplied_dir}/UserSupplied_train_LR_bicubic_X4.zip", None),
    #     # Validation datasets
    #     (f"{usersupplied_dir}/UserSupplied_val_HR.zip", None),
    #     (f"{usersupplied_dir}/UserSupplied_val_LR_bicubic_X8.zip", None),
    #     (f"{usersupplied_dir}/UserSupplied_val_LR_bicubic_X4.zip", None),
    #     # Testing datasets
    #     (f"{usersupplied_dir}/UserSupplied_test_LR_bicubic_X8.zip", None),
    #     (f"{usersupplied_dir}/UserSupplied_test_LR_bicubic_X4.zip", None),
    # ]

    # Track directories for different configurations
    track_dirs = {
        ("hr", "train", 1): os.path.join("UserSupplied_train_HR"),
        ("bicubic", "train", 8): os.path.join("UserSupplied_train_LR_bicubic", "X8"),
        ("bicubic", "train", 4): os.path.join("UserSupplied_train_LR_bicubic", "X4"),
        ("hr", "val", 1): os.path.join("UserSupplied_val_HR"),
        ("bicubic", "val", 8): os.path.join("UserSupplied_val_LR_bicubic", "X8"),
        ("bicubic", "val", 4): os.path.join("UserSupplied_val_LR_bicubic", "X4"),
        ("bicubic", "test", 8): os.path.join("UserSupplied_test_LR_bicubic", "X8"),
        ("bicubic", "test", 4): os.path.join("UserSupplied_test_LR_bicubic", "X4"),
    }

    def __init__(
        self,
        root: str,
        scale: Union[int, List[int], None] = None,
        track: Union[str, List[str]] = "bicubic",
        split: str = "train",
        transform: Optional[Callable] = None,
        loader: Callable = pil_loader,
        download: bool = False,
        predecode: bool = False,
        preload: bool = False,
    ):
        super(UserSupplied, self).__init__(
            os.path.join(root, "UserSupplied"),
            scale,
            track,
            split,
            transform,
            loader,
            download,
            predecode,
            preload,
        )
