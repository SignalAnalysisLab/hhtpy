from typing import Callable
import numpy as np

# mode, total_sifts_performed -> True if stopping criterion is met
SiftStoppingCriterion = Callable[[np.ndarray, int], bool]


def get_stopping_criterion_fixed_number_of_sifts(
    fixed_number_of_sifts: int,
) -> SiftStoppingCriterion:
    def _fixed_number_of_sifts(_, total_sifts_performed: int):
        return total_sifts_performed >= fixed_number_of_sifts

    return _fixed_number_of_sifts
