import os
from typing import Union

def get_seed() -> Union[int, None]:
    seed = os.environ.get("PYHARD_SEED")
    if seed is None:
        return seed
    else:
        try:
            return int(seed)
        except ValueError:
            return None