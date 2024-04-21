# from abc import ABC, abstractmethod
# from collections import defaultdict

# import pandas as pd

# from .rule import Operator, Rule
# from .types import numeric


# class BaseClassifier(ABC):
#     def __init__(
#         self,
#         rules: list[Rule],
#         min_decision_class: numeric | None = None,
#         max_decision_class: numeric | None = None,
#     ) -> None:
#         self.rules = rules

#         self.at_most_rules = defaultdict(list)
#         self.at_least_rules = defaultdict(list)

#         for at_most_rule in [rule for rule in rules if rule.operator == Operator.LE]:
#             self.at_most_rules[at_most_rule.decision].append(at_most_rule)

#         for at_least_rule in [rule for rule in rules if rule.operator == Operator.GE]:
#             self.at_least_rules[at_least_rule.decision].append(at_least_rule)

#         self.at_most_rules = sorted(self.at_most_rules.items(), key=lambda x: x[0])
#         self.at_least_rules = sorted(self.at_least_rules.items(), key=lambda x: x[0])

#         self.min_decision_class = (
#             min_decision_class
#             if min_decision_class is not None
#             else min(min(self.at_least_rules.keys()), min(self.at_most_rules.keys()))
#         )
#         self.max_decision_class = (
#             max_decision_class
#             if max_decision_class is not None
#             else max(max(self.at_least_rules.keys()), max(self.at_most_rules.keys()))
#         )

#     @abstractmethod
#     def classify(self, data): ...


# class SimpleClassifier(BaseClassifier):
#     def _get_classes(self, data: pd.Series, at_most: bool = False) -> list[numeric]:
#         result = []
#         for decision_class, rules in self.at_most_rules.items() if at_most else self.at_least_rules.items():
#             for rule in rules:
#                 if rule.do_match(**data):
#                     result.append(decision_class)
#                     break

#         return result

#     def _get_min_max_recommendation(self, data: pd.Series) -> pd.Series:
#         at_most_recommendation = self._get_classes(data, at_most=True)
#         at_least_recommendation = self._get_classes(data, at_most=False)

#         # 1. only "at least" rules
#         if at_least_recommendation and not at_most_recommendation:
#             return pd.Series(
#                 {
#                     "min": at_least_recommendation[-1],
#                     "max": at_least_recommendation[-1],
#                 }
#             )

#         # 2. only "at most" rules
#         if at_most_recommendation and not at_least_recommendation:
#             return pd.Series(
#                 {
#                     "min": at_most_recommendation[0],
#                     "max": at_most_recommendation[0],
#                 }
#             )

#         # 3. overlapping "at least" and "at most" rules
#         if (
#             at_least_recommendation
#             and at_most_recommendation
#             and at_least_recommendation[0] < at_most_recommendation[-1]
#         ):
#             return pd.Series(
#                 {
#                     "min": at_least_recommendation[0],
#                     "max": at_most_recommendation[-1],
#                 }
#             )

#         # 4. disjoint "at least" and "at most" rules
#         if (
#             at_least_recommendation
#             and at_most_recommendation
#             and at_least_recommendation[0] > at_most_recommendation[-1]
#         ):
#             pd.Series(
#                 {
#                     "min": at_most_recommendation[0],
#                     "max": at_least_recommendation[-1],
#                 }
#             )

#         # 5. no rules
#         return pd.Series(
#             {
#                 "min": self.min_decision_class,
#                 "max": self.max_decision_class,
#             }
#         )

#     def classify(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
#         data = data.copy()
#         if isinstance(data, pd.Series):
#             data.index = [str(c) for c in data.index]

#             return self._get_min_max_recommendation(data)

#         data.columns = [str(c) for c in data.columns]
#         return data.apply(self._get_min_max_recommendation, axis=1)


# class AdvancedClassifier(BaseClassifier):
#     def __init__(self, rules: list[Rule]) -> None:
#         super().__init__(rules)

#     def _score(self, class_value: numeric) -> numeric: ...

#     def _get_recommendation(self, data: pd.Series) -> numeric: ...

#     def classify(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
#         return super().classify(data)
