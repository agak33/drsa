import math
from enum import IntEnum

import pandas as pd
import pytest

from drsa import Alternative, AlternativesSet, Criterion


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
            False,
            pd.Series(
                {
                    "A": {"A", "B", "C", "D", "E", "F", "G", "H", "I"},
                    "B": {"B", "C", "D", "E", "F", "G", "I"},
                    "C": {"C", "D", "E", "F", "G", "I"},
                    "D": {"C", "D", "E", "F", "G", "I"},
                    "E": {"E", "G"},
                    "F": {"F", "G", "I"},
                    "G": {"G"},
                    "H": {"H", "I"},
                    "I": {"I"},
                }
            ),
        ),
        (
            True,
            pd.Series(
                {
                    "A": {"A"},
                    "B": {"A", "B"},
                    "C": {"A", "B", "C", "D"},
                    "D": {"A", "B", "C", "D"},
                    "E": {"A", "B", "C", "D", "E"},
                    "F": {"A", "B", "C", "D", "F"},
                    "G": {"A", "B", "C", "D", "E", "F", "G"},
                    "H": {"A", "H"},
                    "I": {"A", "B", "C", "D", "F", "H", "I"},
                }
            ),
        ),
    ),
)
def test_dominance_cones(negative: bool, expected: pd.Series, alternatives: AlternativesSet):
    assert expected.equals(alternatives.dominance_cones(negative))


@pytest.mark.parametrize(
    ("at_most", "expected"),
    (
        (
            True,
            (
                pd.Series(
                    {
                        Result.C1: {"A", "B"},
                        Result.C2: {"A", "B", "C", "D", "H"},
                        Result.C3: {"A", "B", "C", "D", "F", "H"},
                        Result.C4: {"A", "B", "C", "D", "E", "F", "G", "H", "I"},
                    },
                    name="lower approximation, at most",
                ),
                pd.Series(
                    {
                        Result.C1: {"A", "B", "C", "D"},
                        Result.C2: {"A", "B", "C", "D", "H"},
                        Result.C3: {"A", "B", "C", "D", "E", "F", "G", "H"},
                        Result.C4: {"A", "B", "C", "D", "E", "F", "G", "H", "I"},
                    },
                    name="upper approximation, at most",
                ),
            ),
        ),
        (
            False,
            (
                pd.Series(
                    {
                        Result.C1: {"A", "B", "C", "D", "E", "F", "G", "H", "I"},
                        Result.C2: {"E", "F", "G", "H", "I"},
                        Result.C3: {"E", "F", "G", "I"},
                        Result.C4: {"I"},
                    },
                    name="lower approximation, at least",
                ),
                pd.Series(
                    {
                        Result.C1: {"A", "B", "C", "D", "E", "F", "G", "H", "I"},
                        Result.C2: {"C", "D", "E", "F", "G", "H", "I"},
                        Result.C3: {"E", "F", "G", "I"},
                        Result.C4: {"E", "G", "I"},
                    },
                    name="upper approximation, at least",
                ),
            ),
        ),
    ),
)
def test_class_approximation(
    at_most: bool, expected: tuple[pd.Series, pd.Series], alternatives: AlternativesSet
):
    lower_expected, upper_expected = expected
    lower, upper = alternatives.class_approximation(at_most=at_most)

    assert lower_expected.name == lower.name
    assert upper_expected.name == upper.name

    assert lower_expected.equals(lower)
    assert upper_expected.equals(upper)


@pytest.mark.parametrize(
    ("at_most", "expected"),
    (
        (
            False,
            pd.Series(
                {
                    Result.C1: set(),
                    Result.C2: {"C", "D"},
                    Result.C3: set(),
                    Result.C4: {"E", "G"},
                }
            ),
        ),
        (
            True,
            pd.Series(
                {
                    Result.C1: {"C", "D"},
                    Result.C2: set(),
                    Result.C3: {"E", "G"},
                    Result.C4: set(),
                }
            ),
        ),
    ),
)
def test_boundary(at_most: bool, expected: pd.Series, alternatives: AlternativesSet):
    result = alternatives.boundary(at_most=at_most)

    assert result.name is None or result.name == ""
    assert result.equals(expected)


def test_classification_quality(alternatives: AlternativesSet):
    assert math.isclose(alternatives.classification_quality(), 5 / 9)
