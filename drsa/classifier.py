from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from .rule import Operator, Rule
from .types import numeric


class BaseClassifier(ABC):
    def __init__(
        self,
        rules: list[Rule],
        decision_class_values: list[numeric],
        trained_classifier: ClassifierMixin | None = None,
    ) -> None:
        """Base classifier class in DRSA"""

        self.rules = rules
        self.decision_class_values = sorted(decision_class_values)
        self.trained_classifier = trained_classifier

        self.at_most_rules: dict[numeric, list[Rule]] = defaultdict(list)
        self.at_least_rules: dict[numeric, list[Rule]] = defaultdict(list)

        for at_most_rule in [rule for rule in rules if rule.operator == Operator.LE]:
            self.at_most_rules[at_most_rule.decision].append(at_most_rule)

        for at_least_rule in [rule for rule in rules if rule.operator == Operator.GE]:
            self.at_least_rules[at_least_rule.decision].append(at_least_rule)

        self.at_most_rules = dict(sorted(self.at_most_rules.items(), key=lambda x: x[0]))
        self.at_least_rules = dict(sorted(self.at_least_rules.items(), key=lambda x: x[0]))

    @abstractmethod
    def classify(self, data): ...


class SimpleClassifier(BaseClassifier):
    """Simple classifier, based on the set of matched rules.

    This classifier in its recommendation returns class intervals.
    (minimal and maximal recommended class).
    """

    def _get_classes(self, data: pd.Series, at_most: bool = False) -> list[numeric]:
        result = []
        for decision_class, rules in self.at_most_rules.items() if at_most else self.at_least_rules.items():
            for rule in rules:
                if rule.do_match(**data):
                    result.append(decision_class)
                    break

        return sorted(result)

    def _get_min_max_recommendation(self, data: pd.Series) -> numeric:
        at_most_recommendation = self._get_classes(data, at_most=True)
        at_least_recommendation = self._get_classes(data, at_most=False)

        # 1. only "at least" rules
        if at_least_recommendation and not at_most_recommendation:
            return at_least_recommendation[-1]

        # 2. only "at most" rules
        if at_most_recommendation and not at_least_recommendation:
            return at_most_recommendation[0]

        # 3. overlapping "at least" and "at most" rules
        if (
            at_least_recommendation
            and at_most_recommendation
            and at_least_recommendation[0] < at_most_recommendation[-1]
        ):
            return np.mean([at_least_recommendation[0], at_most_recommendation[-1]]).__float__()

        # 4. disjoint "at least" and "at most" rules
        if (
            at_least_recommendation
            and at_most_recommendation
            and at_least_recommendation[0] > at_most_recommendation[-1]
        ):
            return np.mean([at_most_recommendation[-1], at_least_recommendation[0]]).__float__()

        # 5. no rules - all classes
        return np.mean((self.decision_class_values[0], self.decision_class_values[-1])).__float__()

    def classify(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        data = data.copy()
        if isinstance(data, pd.Series):
            data.index = [str(c) for c in data.index]

            return np.mean(self._get_min_max_recommendation(data))

        data.columns = [str(c) for c in data.columns]
        return data.apply(self._get_min_max_recommendation, axis=1)


class AdvancedClassifier(BaseClassifier):
    def _get_rules_that_covers(self, data: pd.Series) -> list[Rule]:
        return [rule for rule in self.rules if rule.do_match(**data)]

    def _get_positive_score(self, class_value: numeric, rules_set: list[Rule]) -> float:
        objects_in_class = self.rules[0].objects_in_each_class[
            class_value
        ]  # each rule has the same objects_in_each_class

        rules = [
            rule
            for rule in rules_set
            if (rule.decision <= class_value and rule.operator == Operator.GE)
            or (rule.decision >= class_value and rule.operator == Operator.LE)
        ]

        if not rules:
            return 0.0

        return len(set.union(*[rule.conditions.covered_objects & objects_in_class for rule in rules])) ** 2 / (
            len(set.union(*[rule.conditions.covered_objects for rule in rules])) * len(objects_in_class)
        )

    def _get_negative_score(self, class_value: numeric, rules_set: list[Rule]) -> float:
        rules = [
            rule
            for rule in rules_set
            if (rule.decision > class_value and rule.operator == Operator.GE)
            or (rule.decision < class_value and rule.operator == Operator.LE)
        ]

        if not rules:
            return 0.0

        return len(set.union(*[rule.conditions.covered_objects & rule.class_union for rule in rules])) ** 2 / (
            len(set.union(*[rule.conditions.covered_objects for rule in rules]))
            * len(set.union(*[rule.class_union for rule in rules]))
        )

    def _get_score(self, class_value: numeric, rules_set: list[Rule]) -> float:
        return self._get_positive_score(class_value, rules_set) - self._get_negative_score(class_value, rules_set)

    def _get_recommendation(self, data: pd.Series) -> numeric:
        rules = self._get_rules_that_covers(data)
        if not rules:
            return np.mean((self.decision_class_values[0], self.decision_class_values[-1])).__float__()

        if len(rules) == 1:
            return max(rules[0].score, key=lambda key: rules[0].score[key])

        scores = {class_value: self._get_score(class_value, rules) for class_value in self.decision_class_values}
        return max(scores, key=lambda key: scores[key])

    def classify(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        data = data.copy()
        if isinstance(data, pd.Series):
            data.index = [str(c) for c in data.index]

            return self._get_recommendation(data)

        data.columns = [str(c) for c in data.columns]
        return data.apply(self._get_recommendation, axis=1)
