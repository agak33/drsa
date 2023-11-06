import math
from enum import IntEnum

import pandas as pd

from drsa import Alternative, AlternativesSet, Criterion

# class G1(IntEnum):
#     BAD = 1
#     MEDIUM = 2
#     GOOD = 3
#     VERY_GOOD = 4


# class Result(IntEnum):
#     C1 = 1
#     C2 = 2
#     C3 = 3
#     C4 = 4


class Math(IntEnum):
    BAD = 1
    MEDIUM = 2
    GOOD = 3


class Literature(IntEnum):
    BAD = 1
    MEDIUM = 2
    GOOD = 3


class Decision(IntEnum):
    BAD = 1
    MEDIUM = 2
    GOOD = 3


if __name__ == "__main__":
    # criteria = [
    #     Criterion("g1"),
    #     Criterion("g2", is_cost=True),
    #     Criterion("class", is_decision_attr=True),
    # ]

    # alternatives = AlternativesSet(
    #     [
    #         Alternative("A", [G1.BAD, 4, Result.C1], index=criteria),
    #         Alternative("B", [G1.MEDIUM, 4, Result.C1], index=criteria),
    #         Alternative("C", [G1.MEDIUM, 3, Result.C1], index=criteria),
    #         Alternative("D", [G1.MEDIUM, 3, Result.C2], index=criteria),
    #         Alternative("E", [G1.VERY_GOOD, 3, Result.C4], index=criteria),
    #         Alternative("F", [G1.MEDIUM, 2, Result.C3], index=criteria),
    #         Alternative("G", [G1.VERY_GOOD, 2, Result.C3], index=criteria),
    #         Alternative("H", [G1.BAD, 1, Result.C2], index=criteria),
    #         Alternative("I", [G1.GOOD, 1, Result.C4], index=criteria),
    #     ]
    # )
    # criteria = [Criterion("M"), Criterion("L"), Criterion("D", is_decision_attr=True)]

    # alternatives = AlternativesSet(
    #     [
    #         Alternative("S1", [Math.GOOD, Literature.BAD, Decision.BAD], index=criteria),
    #         Alternative("S2", [Math.MEDIUM, Literature.BAD, Decision.MEDIUM], index=criteria),
    #         Alternative("S3", [Math.MEDIUM, Literature.MEDIUM, Decision.MEDIUM], index=criteria),
    #         Alternative("S4", [Math.GOOD, Literature.MEDIUM, Decision.GOOD], index=criteria),
    #         Alternative("S5", [Math.GOOD, Literature.GOOD, Decision.GOOD], index=criteria),
    #         Alternative("S6", [Math.GOOD, Literature.GOOD, Decision.GOOD], index=criteria),
    #         Alternative("S7", [Math.BAD, Literature.BAD, Decision.BAD], index=criteria),
    #         Alternative("S8", [Math.BAD, Literature.MEDIUM, Decision.BAD], index=criteria),
    #     ]
    # )

    # print(alternatives.rules())
    # print(alternatives.rules(at_most=True))

    # criteria = [
    #     FOOD  GENDER  AGE  EDUCATION  INCOME  SPORT  TRAVEL  KNOWLEDGE  RAW SEAFOOD  NOVEL FOOD  TECHNOLOGY  NUTRITION  ENVIRONMENT  CERTIFICATION  CHEF  TASTE  SMELL  CONSISTENCY
    # ]

    a = pd.read_excel("./data/excel database/insects_database.xlsx", sheet_name="Dataset")
    a.drop("Questionnaire no.", axis=1, inplace=True)

    a.rename(
        columns={
            "FOOD": Criterion("Food", is_decision_attr=True),
            "GENDER": Criterion("Gender", is_cost=True),
            "AGE": Criterion("Age", is_cost=True),
            "EDUCATION": Criterion("Education"),
            "INCOME": Criterion("Income", is_cost=True),
            "SPORT": Criterion("Sport"),
            "TRAVEL": Criterion("Travel"),
            "KNOWLEDGE": Criterion("Knowledge"),
            "RAW SEAFOOD": Criterion("Raw Seafood"),
            "NOVEL FOOD": Criterion("Novel Food"),
            "TECHNOLOGY": Criterion("Technology"),
            "NUTRITION": Criterion("Nutrition"),
            "ENVIRONMENT": Criterion("Environment"),
            "CERTIFICATION": Criterion("Certification"),
            "CHEF": Criterion("Chef"),
            "TASTE": Criterion("Taste"),
            "SMELL": Criterion("Smell"),
            "CONSISTENCY": Criterion("Consistency"),
        },
        inplace=True,
    )

    data = AlternativesSet(a)

    print(data)

    print(data.rules())
