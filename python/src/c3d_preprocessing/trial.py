from dataclasses import dataclass


@dataclass
class Trial:
    participant: str
    name: str
    fp_left: int
    fp_right: int
    start_time: float
    end_time: float
