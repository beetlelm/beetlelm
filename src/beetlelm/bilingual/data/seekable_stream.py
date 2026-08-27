#beetlelm/src/bilingual/data/seekable_stream.py

from typing import Iterator


class SeekableTextStream:
    """
    Deterministic streaming text iterator with checkpoint support.
    Works for line-based corpora.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, "r", encoding="utf-8")
        self.position = 0

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self):
        line = self.file.readline()
        if line == "":
            self.file.seek(0)
            line = self.file.readline()

        self.position = self.file.tell()
        return line.strip()

    # ----- Checkpoint -----

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.file.seek(state["position"])
        self.position = state["position"]
