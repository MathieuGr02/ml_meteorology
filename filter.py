import json
import os
import subprocess
from pathlib import Path

"""
This script if for the applying of `removable_keys.json` onto the raw downloaded data
"""


def filter_data(file: str):
    removable_keys = []
    with open("removable_keys.json", "r") as f:
        data = json.load(f)
        removable_keys = data["keys"]

    subprocess.run(
        [
            "ncks",
            "-x",
            "-v",
            ",".join(removable_keys),
            f"./rawdata/{file}.nc4",
            f"./data/{file}_filtered.nc4",
        ]
    )


if __name__ == "__main__":
    for file in os.listdir("./rawdata"):
        name_w_ext = os.path.basename(file)
        name_wo_ext = Path(name_w_ext).stem
        filter_data(name_wo_ext)
