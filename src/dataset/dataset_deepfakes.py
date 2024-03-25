import json
import math
import random
import time
from ast import Call
from pathlib import Path
from re import S
from typing import Any, Callable, Dict, List, Optional, Tuple


import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from src.utils.bbox import bboxes_intersection_over_union
from src.utils.dataframe import convert_partial_int
from src.utils.logger import format_time
from src.utils.omega import omegaconf_to_dict, parse_selector
from torch.utils.data import Dataset

from .face_selector import select_principal_face


def retrive_data(filename: Path) -> Any:
    if not filename.is_file():
        return
    if filename.suffix == ".json":
        with filename.open("r") as f:
            data = json.load(f)
    elif filename.suffix == ".npy":
        data = np.load(str(filename))
    else:
        with filename.open("r") as f:
            data = f.read()
    return data


class DatasetDeepfakes(Dataset):
    """
    Base class for deepfakes datasets using a csv file.
    Each row represents a video
    path, label, fold, split

    torchvision/datasets/folder.py
    .jpg,.jpeg,.png,.ppm,.bmp,.pgm,.tif,.tiff,.webp
    """

    def __init__(
        self,
        root: str,
        csv: str,
        name: Optional[str] = None,
        selector: Optional[DictConfig] = None,
        metas: Optional[DictConfig] = None,
        level: Optional[str] = None,
        use_cache: bool = True,
        transform: Optional[Callable] = None,
    ):        
        self.root = Path(root)
        if not self.root.is_dir():
            # raise ValueError(f"{self.root} is not a directory")
            logger.info(f"{self.root} is not a directory")
        self.path_csv = Path(csv)
        if not self.path_csv.is_file():
            raise ValueError(f"{self.path_csv} is not a file")
        self.name = name
        self.transform = transform

        # read the full dataframe from the csv
        self._df: pd.DataFrame = pd.read_csv(csv)
        # self._df = convert_partial_int(self._df)

        # convert omegaconf.DictConfig to dict
        self.metas_dict = omegaconf_to_dict(metas)
        self.selector: Dict = omegaconf_to_dict(selector)
        self.update_selector(self.selector)

        # folder where additional frames' data are stored
        # self.dirs = [self.root / "retina"]
        if 1:
            self.dirs = [
                p for p in self.root.iterdir() if (p.is_dir() and p.name != "images")
            ]
        else:
            self.dirs = [self.root / "retina"]

        # level of the dataset
        if level == "":
            level = None
        assert level in [
            None,
            "frame",
            "face",
        ], f"level must be in [None, 'frame', 'face'], got {level}"
        self.level = level

        # cache related
        self._cache = dict()
        self.path_cache = self.root / "cache.json"
        load_cache = use_cache and level in ["frame", "face"]
        if load_cache:
            self._cache = self.load_or_generate_cache(
                self.path_cache, force_generate=False
            )

    def update_selector(self, selector: Dict, deep: bool = False):
        """take a dict, parse it and get a selector function
        that returns a boolean mask array.
        Then use this selector to subset the full private _df to a public df
        """
        F = parse_selector(selector)
        assert callable(F), f"selector must be a callable, got {type(F)}"
        indexs_mask: np.ndarray = F(self._df)

        c = indexs_mask.sum()
        t = len(indexs_mask)
        p = round(100 * c / t, 2)
        logger.info(f"selector: {selector} -> {c} / {t} ({p}%)")

        # partial view of the dataset
        self.df: pd.DataFrame = self._df[indexs_mask].copy(deep=deep)

        # update related attributes
        self.selected_indexs: np.ndarray = np.where(indexs_mask)[0]
        self.targets = self.df["label"].values
        self.real_mask = self.targets == 0
        self.fake_mask = ~self.real_mask

    # cache related methods ----------------------------------------------------

    def retrieve_frames_data(self, index: int, ext: str = ".png") -> Dict:
        """Retrieve data from a list of frames index
        Folder structure:
            images|0|0.png
                    |27.png
            metas |0.txt
            retina|0|0.json
                    |27.json
        Input:
            index: int; ie 0
        Output:
            {"0": {"metas": "I", "retina": [{}], ...},
             "27": {"metas": "I", "retina": [{}], ...}}
        """
        # retrieve frames
        frames_path = (self.root / f"images/{index}").glob(f"*{ext}")
        frames_index = sorted([int(p.stem) for p in frames_path])
        if not frames_index:
            return {}

        out = {f: dict() for f in frames_index}
        dirs = [d for d in self.dirs if d.is_dir()]
        for dir in dirs:
            name = dir.name
            name_folder = dir / str(index)
            name_txt = dir / f"{index}.txt"
            if name_folder.is_dir():
                for f in frames_index:
                    data = retrive_data(name_folder / f"{f}.json")
                    if data is not None:
                        out[f][name] = data
            elif name_txt.is_file():
                data = retrive_data(name_txt).split(",")
                for f in frames_index:
                    if f < len(data):
                        out[f][name] = data[f]
            elif name == "videos":
                continue
            else:
                raise ValueError(f"no data found for {dir.name}/{index}")

        return out

    def generate_cache(self, path: Path):
        # images folder where frames are stored
        dir_images = self.root / "images"
        assert dir_images.is_dir(), f"{dir_images} is not a directory"

        # option 1
        indexs = range(len(self._df))
        # option 2: only cache selected indexs
        # indexs = list(map(int, self.selected_indexs))

        cache = {}
        for index in indexs:
            # cached = {f: dict() for f in frames_index}
            cache[index] = self.retrieve_frames_data(index)

        logger.info("Saving cache ...")
        cache_json = json.dumps(cache)
        # save as json
        with path.open("w") as f:
            f.write(cache_json)

        return cache

    def load_or_generate_cache(self, path: Path, force_generate: bool = False) -> Dict:
        logger.info(f"searching cache from {path}")
        if path.is_file() and not force_generate:
            logger.info("Loading cache ...")
            with path.open("r") as f:
                cache = json.load(f)
        else:
            logger.info("Generating cache ...")
            t0 = time.time()
            cache = self.generate_cache(path)
            logger.info(f"cache generated in {format_time(time.time() - t0)}")
        return cache

    # special methods ----------------------------------------------------------

    def _getitem(self, idx: int, resolve_path: bool = False) -> Tuple[int, Dict]:
        """Return a sample (row) of the public dataframe"""
        # tmp
        # df.where(pd.notnull(df), None)
        sample = self.df.iloc[idx].to_dict()

        index = int(self.selected_indexs[idx])
        sample["index"] = index
        if resolve_path:
            sub_path = sample["path"].format(**self.metas_dict)
            sample["path"] = str(self.root / sub_path)
        return index, sample

    def _getitem_frames(self, idx: int) -> List[Dict]:
        """return same dict as getitem, but with a key "frames"
        which will contains a list of dict of the frames
        ie;
        {"frames":[
            {"index":0 , "path":., "meta": "I", "retina":{ }, "dlib68":{ }},
            {"index":34, "path":., "meta": "I", "retina":{ }, "dlib68":{ }},
        ]}
        """
        index, sample = self._getitem(idx, resolve_path=False)

        # retrieve frames by cache or by reading the folder
        frames = self._cache.get(str(index))
        if frames is None:
            frames = self.retrieve_frames_data(index)
            assert frames, f"no frames found for {index}"
            # raise ValueError(f"no cache found for {index}")

        frames_list = []
        for frame_idx, frame_data in frames.items():
            frame_path = self.root / f"images/{index}/{frame_idx}.png"
            assert frame_path.is_file(), f"{frame_path} is not a file"
            frame_data["path"] = str(frame_path)
            frame_data["index"] = frame_idx
            frames_list.append(frame_data)

        out = dict()
        out["video"] = sample
        out["label"] = sample["label"]
        out["index"] = sample["index"]
        out["frames"] = frames_list
        return out

    def _getitem_face(self, idx: int) -> Dict:
        """for a given video at idx, select a random frame
        and return a dict with the face"""
        sample = self._getitem_frames(idx)
        video = sample["video"]
        frames = sample["frames"]
        width = video.get("frame_width")
        height = video.get("frame_height")
        n_frames = len(frames)

        # We work in a video level: we select a random frame from the video
        remaining_frame_index = list(range(n_frames))
        np.random.shuffle(remaining_frame_index)

        # we try to find a frame with a face-landmarks-bbox
        face_data = None
        while face_data is None and len(remaining_frame_index) > 0:
            frame_idx = remaining_frame_index.pop()
            frame = frames[frame_idx]
            try:
                face_data = select_principal_face(frame, width=width, height=height)
                # face_data = None
            except Exception as e:
                # face_data = None
                print(e)
                logger.error(
                    "no face found for "
                    + f"{sample['video']['index']} / {frame['index']} "
                    + frame["path"],
                    exc_info=e,
                )

        # if no face found, we return the video
        # otherwise, face_data and frame are usable
        if face_data is None and len(remaining_frame_index) == 0:
            raise ValueError(f"no face found for {sample['video']['index']}")

        # post process
        path = frame["path"]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        assert isinstance(image, np.ndarray), f"failed to read {path}"
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out = dict()
        out["video"] = video  # video_level
        out["frame"] = frame  # frame_level
        out["label"] = video["label"]
        out["index"] = video["index"]
        out["face"] = face_data
        out["image"] = image
        return out

    def __getitem__(self, idx: int) -> Dict:
        assert isinstance(idx, int), f"idx must be an int, not {type(idx)}"
        if self.level is None:
            index, sample = self._getitem(idx, resolve_path=False)
        elif self.level == "frame":
            sample = self._getitem_frames(idx)
        elif self.level == "face":
            sample = self._getitem_face(idx)
        else:
            ValueError(f"level {self.level} is not supported")

        # transform = self.transform
            
        # if self.transform is not None:
        #     # sample = self.transform(sample=sample)
        #     # logger.info(str(sample))
        #     sample = self.transform(**sample)
            
        return sample

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        repr = [f"{self.__class__.__name__}(root={self.root}, csv={self.path_csv})"]
        return "\n".join(repr)
