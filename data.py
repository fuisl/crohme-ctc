import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

import glob
import os

import numpy as np
import xml.etree.ElementTree as ET


class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""

    __slots__ = ("id", "label", "strId")

    def __init__(self, *args):
        if len(args) == 3:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""

    __slots__ = ("fileName", "strokes", "strkOrder", "segments", "truth", "UI")

    NS = {
        "ns": "http://www.w3.org/2003/InkML",
        "xml": "http://www.w3.org/XML/1998/namespace",
    }

    def __init__(self, *args):
        self.fileName = None
        self.strokes = {}
        self.strkOrder = []
        self.segments = {}
        self.truth = ""
        self.UI = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()

    def fixNS(self, ns, att):
        """Build the right tag or element name with namespace"""
        return "{" + Inkml.NS[ns] + "}" + att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall("ns:annotation", namespaces=Inkml.NS):
            if "type" in info.attrib:
                if info.attrib["type"] == "truth":
                    self.truth = info.text.strip()
                if info.attrib["type"] == "UI":
                    self.UI = info.text.strip()
        for strk in root.findall("ns:trace", namespaces=Inkml.NS):
            self.strokes[strk.attrib["id"]] = strk.text.strip()
            self.strkOrder.append(strk.attrib["id"])
        segments = root.find("ns:traceGroup", namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            return
        for seg in segments.iterfind("ns:traceGroup", namespaces=Inkml.NS):
            id = seg.attrib[self.fixNS("xml", "id")]
            label = seg.find("ns:annotation", namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall("ns:traceView", namespaces=Inkml.NS):
                strkList.add(t.attrib["traceDataRef"])
            self.segments[id] = Segment(id, label, strkList)

    def getTraces(self, height=256):
        traces_array = [
            np.array(
                [p.strip().split() for p in self.strokes[id].split(",")], dtype="float"
            )
            for id in self.strkOrder
        ]

        ratio = height / (
            (
                np.concatenate(traces_array, 0).max(0)
                - np.concatenate(traces_array, 0).min(0)
            )[1]
            + 1e-6
        )
        return [(trace * ratio).astype(int).tolist() for trace in traces_array]


"""
TODO: 
- [x] build vocabulary
- [x] remove duplicate (next in sequence) --> no duplicate found in the dataset
- [x] normalize (delta x, delta y)
- [x] concat traces in a single array (n, 2)
- [x] pad to max length (collate) ==> do in Dataloader
"""


class InkmlDataset(Dataset):
    def __init__(self, annotation, root_dir="dataset"):
        with open(annotation, "r") as f:
            self.files = f.readlines()

        self.vocab = {
            "": 0,
            "-": 1,
            "\\times": 2,
            "\\{": 3,
            "\\beta": 4,
            "m": 5,
            "Above": 6,
            "E": 7,
            "\\infty": 8,
            "\\forall": 9,
            "\\cos": 10,
            "8": 11,
            ")": 12,
            "/": 13,
            "\\sum": 14,
            "n": 15,
            "\\pi": 16,
            "\\geq": 17,
            "C": 18,
            "a": 19,
            "\\mu": 20,
            "S": 21,
            "]": 22,
            "R": 23,
            "\\gt": 24,
            "Sup": 25,
            "x": 26,
            "p": 27,
            "\\ldots": 28,
            "\\int": 29,
            "\\sqrt": 30,
            "f": 31,
            "Right": 32,
            "k": 33,
            "\\log": 34,
            "\\leq": 35,
            "j": 36,
            "w": 37,
            "7": 38,
            "y": 39,
            "\\exists": 40,
            "d": 41,
            "[": 42,
            "q": 43,
            "\\div": 44,
            "NoRel": 45,
            "\\phi": 46,
            "1": 47,
            "g": 48,
            "X": 49,
            "\\in": 50,
            "\\gamma": 51,
            "\\prime": 52,
            "4": 53,
            "\\pm": 54,
            "T": 55,
            "F": 56,
            "N": 57,
            "\\lt": 58,
            "o": 59,
            "u": 60,
            "h": 61,
            "s": 62,
            "6": 63,
            "c": 64,
            "(": 65,
            "A": 66,
            "!": 67,
            "P": 68,
            "L": 69,
            "COMMA": 70,
            "i": 71,
            "b": 72,
            "t": 73,
            "+": 74,
            "\\neq": 75,
            "9": 76,
            "3": 77,
            "G": 78,
            ".": 79,
            "e": 80,
            "M": 81,
            "r": 82,
            "\\sin": 83,
            "\\lim": 84,
            "\\lambda": 85,
            "I": 86,
            "\\rightarrow": 87,
            "Inside": 88,
            "\\sigma": 89,
            "V": 90,
            "\\theta": 91,
            "l": 92,
            "=": 93,
            "\\tan": 94,
            "z": 95,
            "2": 96,
            "H": 97,
            "0": 98,
            "5": 99,
            "Below": 100,
            "|": 101,
            "\\Delta": 102,
            "\\alpha": 103,
            "B": 104,
            "Y": 105,
            "v": 106,
            "\\}": 107,
            "Sub": 108,
        }

        self.root_dir = root_dir
        self.inks = []
        self.labels = []

        for line in self.files:
            path, label = line.strip("\n").split("\t")
            inkml = Inkml(os.path.join(self.root_dir, path))

            self.inks.append(inkml)
            self.labels.append(label)

            # traces = inkml.getTraces()
            # for trace in traces:
            #     for i in range(0, len(trace) - 1):
            #         if trace[i] == trace[i + 1]:
            #             print(i)

    def __len__(self):
        return len(self.inks)

    def __getitem__(self, idx):
        traces = self.inks[idx].getTraces()
        combined_traces = np.vstack([np.array(trace)[:, :2] for trace in traces])
        delta_traces = np.diff(combined_traces, axis=0)
        zeros_filter = np.all(delta_traces == 0, axis=1)
        delta_traces = delta_traces[~zeros_filter]
        delta_traces = delta_traces / np.sqrt((np.square(delta_traces[:, 0]) + np.square(delta_traces[:, 1])))[:, np.newaxis]  # delta x, delta y --> delta x/sqrt(delta x^2 + delta y^2), delta y/sqrt(delta x^2 + delta y^2

        pen_up = [np.array([0] * len(trace)) for trace in traces]
        for _, arr in enumerate(pen_up):
            arr[0] = 1

        combined_pen_up = np.concatenate(pen_up)[1:, np.newaxis][~zeros_filter]
        combined_traces = np.hstack([delta_traces, combined_pen_up])
        delta_traces_tensor = torch.tensor(combined_traces, dtype=torch.float32)

        translated_label = [self.vocab[label] for label in self.labels[idx].split(" ")]
        label_tensor = torch.tensor(translated_label, dtype=torch.long)

        return (
            delta_traces_tensor,
            label_tensor,
            combined_traces.shape[0],
            len(translated_label),
        )


class InkmlDataset_PL(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 10,
        workers: int = 5,
        train_data: str = "dataset/crohme2019_train.txt",
        val_data: str = "dataset/crohme2019_valid.txt",
        test_data: str = "dataset/crohme2019_test.txt",
        root_dir: str = "dataset",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.root_dir = root_dir

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = InkmlDataset(self.train_data, root_dir=self.root_dir)
            self.val_dataset = InkmlDataset(self.val_data, root_dir=self.root_dir)
        if stage == "test" or stage is None:
            self.test_dataset = InkmlDataset(self.test_data, root_dir=self.root_dir)

    def custom_collate_fn(self, data):
        traces, labels, len_traces, len_labels = zip(*data)
        padded_traces = pad_sequence(traces, batch_first=True)
        padded_labels = pad_sequence(labels, batch_first=True)
        return padded_traces, padded_labels, len_traces, len_labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.custom_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.custom_collate_fn,
        )


if __name__ == "__main__":
    # dataset = InkmlDataset("dataset/crohme2019_test.txt")
    # dataset.__getitem__(0)
    # print(len(dataset))

    dm = InkmlDataset_PL()
    dm.setup(stage="test")
    data_loader = dm.test_dataloader()
    item = next(iter(data_loader))
    print(item[0].shape, item[1].shape)
