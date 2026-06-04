import numpy as np
from numpy.typing import NDArray

class Index:
    def add(self, x: NDArray[np.float32]) -> None: ...

class IndexFlatL2(Index):
    def __init__(self, d: int) -> None: ...
