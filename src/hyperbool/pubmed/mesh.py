import os
from pathlib import Path
from typing import List

import appdirs
import lucene

from hyperbool import util
from hyperbool.pubmed.datautils import MESH_YEAR, MESH_URL

DEFAULT_PATH = Path(appdirs.user_data_dir("hyperbool")) / "data/mesh"


def analyze_mesh(heading: str) -> str:
    return heading.lower(). \
        strip(). \
        replace("-", " "). \
        replace("+", " "). \
        replace(")", " "). \
        replace("(", " ")


class MeSHTree:
    def __init__(self, mtrees_file: Path = DEFAULT_PATH, year: str = MESH_YEAR):
        self.locations = {}
        self.headings = []
        if not Path(mtrees_file / f"mtrees{year}.bin").exists():
            download_mesh()
        with open(mtrees_file / f"mtrees{year}.bin", "r") as f:
            for i, line in enumerate(f):  # Assumes all headings are sorted in order of location.
                heading, location = line.replace("\n", "").strip().split(";")
                # TODO: Need to index mesh headings in the same way.
                analyzed_heading = analyze_mesh(heading)
                # analyzed_heading = heading
                self.locations[analyzed_heading] = i
                self.headings.append((location.strip(), heading))

    def explode(self, heading: str) -> List[str]:
        analyzed_heading = analyze_mesh(heading)
        if analyzed_heading not in self.locations:
            return []
        index = self.locations[analyzed_heading]
        exploded_location, exploded_heading = self.headings[index]
        for indexed_heading in self.headings[index:]:
            location, heading = indexed_heading
            if location.startswith(exploded_location):
                yield heading

    def map_heading(self, heading: str) -> str:
        analyzed_heading = analyze_mesh(heading)
        if analyzed_heading not in self.locations:
            return heading
        index = self.locations[analyzed_heading]
        _, found_heading = self.headings[index]
        return str(found_heading).strip()


def download_mesh(path: Path = DEFAULT_PATH, year: str = MESH_YEAR) -> None:
    os.makedirs(str(path), exist_ok=True)
    remote_fname = f"mtrees{year}.bin"
    util.download_file(f"{MESH_URL}{remote_fname}", path / remote_fname)


def exists(path: Path = DEFAULT_PATH, year: str = MESH_YEAR) -> bool:
    return os.path.exists(path / f"mtrees{year}.bin")
