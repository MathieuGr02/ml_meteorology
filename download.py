from time import sleep
from dotenv import load_dotenv
import os
import requests
from pathlib import Path


"""
This script is for the downloading of the data
"""

load_dotenv()


def request_data(url: str, file: str):
    if os.path.exists(file):
        return
    print("URL: ", url, "FILE: ", file)
    # Required Personal Account Authorization token
    headers = {"Authorization": f"Bearer {os.getenv('TOKEN')}"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(response.status_code, response.content)
        quit()

    # Write response content to rawdata file
    print("Writing content")
    with open(file, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    print("Reading file")

    # Summary files of required data
    training_data_meta_path = (
        "subset_AIRX3STD_7.0_20251015_153006_20150104-20153004.txt"
    )
    test_data_meta_path = "subset_AIRX3STD_7.0_20251015_153228_20160104-20163004.txt"

    for meta_path in [training_data_meta_path, test_data_meta_path]:
        with open(meta_path, "r") as file:
            for line in file.readlines():
                for sub_line in line.split("\n"):
                    sub_line = sub_line.strip()
                    # First two lines of summary file are the manuals
                    if sub_line.endswith(".pdf") or sub_line == "":
                        continue

                    sub_line = sub_line.split("?")
                    url, rest = sub_line[0], sub_line[1:]

                    file = f"./rawdata/{url.split('/')[-1]}"

                    if Path(file).exists():
                        continue

                    request_data(url, file)
                    sleep(1)
