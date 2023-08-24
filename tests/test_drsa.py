import pytest
from drsa import Alternative, Criterion, AlternativesSet
from enum import IntEnum
import pandas as pd


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


@pytest.fixture
def criteria() -> list[Criterion]:
    return [
        Criterion("g1"),
        Criterion("g2", is_cost=True),
        Criterion("class", is_decision_attr=True),
    ]


@pytest.fixture
def alternatives(criteria) -> AlternativesSet:
    return AlternativesSet(
        [
            Alternative("A", [G1.BAD, 4, Result.C1], index=criteria),
            Alternative("B", [G1.MEDIUM, 4, Result.C1], index=criteria),
            Alternative("C", [G1.MEDIUM, 3, Result.C1], index=criteria),
            Alternative("D", [G1.MEDIUM, 3, Result.C2], index=criteria),
            Alternative("E", [G1.VERY_GOOD, 3, Result.C4], index=criteria),
            Alternative("F", [G1.MEDIUM, 2, Result.C3], index=criteria),
            Alternative("G", [G1.VERY_GOOD, 2, Result.C3], index=criteria),
            Alternative("H", [G1.BAD, 1, Result.C2], index=criteria),
            Alternative("I", [G1.GOOD, 1, Result.C4], index=criteria),
        ]
    )


@pytest.mark.parametrize(
    ("negative", "expected"),
    (
        (
            True,
            pd.Series(
                {
                    "A": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                    "B": ["B", "C", "D", "E", "F", "G", "I"],
                    "C": ["C", "D", "E", "F", "G", "I"],
                    "D": ["C", "D", "E", "F", "G", "I"],
                    "E": ["E", "G"],
                    "F": ["F", "G", "I"],
                    "G": ["G"],
                    "H": ["H", "I"],
                    "I": ["I"],
                }
            ),
        ),
        (
            False,
            pd.Series(
                {
                    "A": ["A"],
                    "B": ["A", "B"],
                    "C": ["A", "B", "C", "D"],
                    "D": ["A", "B", "C", "D"],
                    "E": ["A", "B", "C", "D", "E"],
                    "F": ["A", "B", "C", "D", "F"],
                    "G": ["A", "B", "C", "D", "E", "F", "G"],
                    "H": ["A", "H"],
                    "I": ["A", "B", "C", "D", "F", "H", "I"],
                }
            ),
        ),
    ),
)
def test_dominance_cones(negative: bool, expected: pd.Series, alternatives: AlternativesSet):
    assert expected.equals(alternatives.dominance_cones(negative))
