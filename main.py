from drsa.drsa import Criterion, Alternative, AlternativesSet
from enum import IntEnum


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


if __name__ == "__main__":
    criteria = [Criterion("g1"), Criterion("g2"), Criterion("class", is_decision_attr=True)]
    data = AlternativesSet(
        [
            Alternative("A", [G1.BAD, 1, Result.C1], index=criteria),
            Alternative("B", [G1.MEDIUM, 1, Result.C1], index=criteria),
            Alternative("C", [G1.MEDIUM, 2, Result.C1], index=criteria),
            Alternative("D", [G1.MEDIUM, 2, Result.C2], index=criteria),
            Alternative("E", [G1.VERY_GOOD, 2, Result.C4], index=criteria),
            Alternative("F", [G1.MEDIUM, 3, Result.C3], index=criteria),
            Alternative("G", [G1.VERY_GOOD, 3, Result.C3], index=criteria),
            Alternative("H", [G1.BAD, 4, Result.C2], index=criteria),
            Alternative("I", [G1.GOOD, 4, Result.C4], index=criteria),
        ]
    )

    print(data.dominance_cones())
    print(data.dominance_cones(negative=True))

    print(data.class_approximation())
    print(data.class_approximation(at_most=True))
