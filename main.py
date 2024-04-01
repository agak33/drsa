from enum import IntEnum

import pandas as pd

from drsa.alternative_set import AlternativesSet
from drsa.criterion import Criterion, DecisionCriterion


class G1(IntEnum):
    BAD = 1
    MEDIUM = 2
    GOOD = 3
    VERY_GOOD = 4


class Result(IntEnum):
    C1 = 1
    C2 = 2
    C3 = 3
    C4 = 4


criteria = [
    Criterion("g1"),
    Criterion("g2", is_cost=True),
    DecisionCriterion("class"),
]


data = AlternativesSet(
    df=pd.DataFrame(
        [
            [G1.BAD, 4, Result.C1],
            [G1.MEDIUM, 4, Result.C1],
            [G1.MEDIUM, 3, Result.C1],
            [G1.MEDIUM, 3, Result.C2],
            [G1.VERY_GOOD, 3, Result.C4],
            [G1.MEDIUM, 2, Result.C3],
            [G1.VERY_GOOD, 2, Result.C3],
            [G1.BAD, 1, Result.C2],
            [G1.GOOD, 1, Result.C4],
        ],
        index=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        columns=criteria,
    )
)

# print(
#     pd.concat(
#         [
#             data[data.cost_attributes].loc["A"] > data[data.cost_attributes],
#             data[data.cost_attributes].loc["A"] > data[data.cost_attributes],
#         ],
#         axis=1,
#     )
# )


data.dominance_cones()

print(data.negative_cones)
print(data.positive_cones)

data.class_unions()

print(data.at_least_approximations)
print(data.at_most_approximations)
