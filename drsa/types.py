from enum import Enum
from typing import Any, Iterable

numeric = int | float


class DominanceType(Enum):
    DOMINATING = "dominating"
    DOMINATED = "dominated"


class Dominance:
    def __init__(
        self,
        object_name: str,
        objects: Iterable,
        dominance_type: DominanceType,
        object_class: Any,
        min_class: Any = None,
        max_class: Any = None,
    ) -> None:
        self.object_name = object_name
        self.objects = objects
        self.dominance_type = dominance_type
        self.object_class = object_class
        self.min_class = min_class
        self.max_class = max_class

    def __repr__(self) -> str:
        return {
            "object_name": self.object_name,
            "objects": self.objects,
            "dominance_type": self.dominance_type,
            "object_class": self.object_class,
            "min_class": self.min_class,
            "max_class": self.max_class,
        }.__repr__()


class ApproximationType(Enum):
    AT_MOST = "at_most"
    AT_LEAST = "at_least"


class ClassApproximation:
    def __init__(
        self,
        class_value: Any,
        approximation_type: ApproximationType,
        lower_approximation: list | set,
        upper_approximation: list | set,
        positive_region: list | set,
        class_union: list | set,
    ) -> None:
        self.class_value = class_value
        self.approximation_type = approximation_type

        self.accuracy = None
        self.quality = None

        self.lower_approximation = lower_approximation
        self.upper_approximation = upper_approximation
        self.positive_region = positive_region

        self.boundary = self._calculate_boundary(lower_approximation, upper_approximation)
        self.quality = self._calculate_quality(class_union)
        self.accuracy = self._calculate_accuracy()

    def __repr__(self) -> str:
        return {
            "class_value": self.class_value,
            "approximation_type": self.approximation_type,
            "lower_approximation": self.lower_approximation,
            "upper_approximation": self.upper_approximation,
            "positive_region": self.positive_region,
            "boundary": self.boundary,
            "quality": self.quality,
            "accuracy": self.accuracy,
        }.__repr__()

    def _calculate_boundary(self, lower_approximation, upper_approximation):
        return list(set(upper_approximation) - set(lower_approximation))

    def _calculate_accuracy(self) -> float:
        return 0.0 if not self.upper_approximation else len(self.lower_approximation) / len(self.upper_approximation)

    def _calculate_quality(self, class_union) -> float:
        return 0.0 if not class_union else len(self.lower_approximation) / len(class_union)
