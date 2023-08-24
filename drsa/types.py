from enum import Enum

numeric = int | float


class RelationType(Enum):
    INDIFFERENT = "INDIFFERENT"
    DOMINATING = "DOMINATING"
    DOMINATED = "DOMINATED"
    INCOMPARABLE = "INCOMPARABLE"
