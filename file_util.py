import json
import pandas as pd
from pathlib import Path

# Save dataset to file
def save_data_to_file(filename, data):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)
        f.write("\n")
    return

# Read dataset from file
def read_data_from_file(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data
