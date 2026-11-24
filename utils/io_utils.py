from pathlib import Path
import pickle
import os
from typing import Any

def serialize_to_file(
    filename: str,
    object: Any,
    force: bool = False
):
    if not force:
        file = Path(filename)
        if file.exists():
            answer = input(
                f"{filename} already exists, overwrite? [y/N] (N to provide a new name)"
            ).strip()
            if answer.lower() == 'y':
                serialize_to_file(filename, object, True)
                return
            elif len(answer) == 0 or answer.lower() == 'n':
                new_path = input("new path to save to: ")
                serialize_to_file(new_path, object, False)
                return
    with open(
        file = filename,
        mode = 'wb'
    ) as f:
        pickle.dump(object, f)
    return

def load_from_file(
    filename: str
) -> Any:
    with open(
        file = filename,
        mode = 'rb',
    ) as f:
        return pickle.load(f)