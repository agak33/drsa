from enum import Enum

from .criterion import Criterion
from .types import numeric


class Operator(Enum):
    GE = ">="
    LE = "<="


class RulesType(Enum):
    CERTAIN = "certain"
    APPROXIMATE = "approximate"
    POSSIBLE = "possible"


class Condition:
    def __init__(
        self,
        field: Criterion,
        operator: Operator,
        value: numeric,
        covered_objects: set | list | None = None,
        all_objects_g: set | list | None = None,
    ) -> None:
        """Represents a single condition within a rule, including a criterion (field), an operator,
        a numeric value, and the set of objects covered by this condition.

        :param field: The criterion of the condition.
        :param operator: The operator used to compare the criterion with a value.
        :param value: The numeric value to compare against the criterion.
        :param covered_objects: The set of objects that are covered by this condition.
        """
        self.field = field
        self.operator = operator
        self.value = value
        self.covered_objects = [] if covered_objects is None else covered_objects

        self.condition_str = f"{field} {operator.value} {value}"
        self.cover_factor = 0.0

        if covered_objects is not None and all_objects_g is not None:
            self.cover_factor = len(set(covered_objects) & set(all_objects_g)) / len(covered_objects)

    def __hash__(self) -> int:
        return hash(self.condition_str)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Condition):
            return NotImplemented
        return self.condition_str == __value.condition_str

    def __repr__(self) -> str:
        return self.condition_str


class Conjunction:
    def __init__(self, conditions: list[Condition] | None = None):
        """Represents a conjunction of conditions that collectively define a rule. This class manages
        the aggregation of individual conditions, their combined coverage, and compiles them into an
        executable rule.
        """
        if conditions is None:
            conditions = []
        self.conditions: list[Condition] = conditions if conditions else []
        self.covered_objects = self._calculate_covered_objects()

        self._compiled_conjunction = lambda *args, **kwargs: True
        self._compiled_conjunction_str = "<empty conjunction>"

    def _calculate_covered_objects(self) -> set:
        """Calculates the set of objects covered by the conjunction of conditions."""
        if not self.conditions:
            return set()

        return set.intersection(*[set(condition.covered_objects) for condition in self.conditions])

    def _compile_rule(self, with_lambda: bool = True) -> None:
        """Compiles the conjunction of conditions into an executable rule. This method
        generates a lambda function and its string representation based on the current
        conditions.
        """
        try:
            self._compiled_conjunction_str = " and ".join([condition.condition_str for condition in self.conditions])
            if with_lambda:
                self._compiled_conjunction = eval(
                    f"lambda {','.join([condition.field.__str__() for condition in self.conditions])}, **kwargs:"
                    f" {' and '.join([condition.condition_str for condition in self.conditions])}"
                )
        except Exception as e:
            raise ValueError(f"Error during rule compilation ({self._compiled_conjunction_str}): {e}")

    def __repr__(self) -> str:
        return self._compiled_conjunction_str

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Conjunction):
            return NotImplemented
        return set(self.conditions) == set(__value.conditions)

    def __add__(self, condition: Condition) -> "Conjunction":
        """Adds a new condition to the conjunction, updates the covered objects and re-compiles the rule."""
        if not self.conditions:
            self.covered_objects = set(condition.covered_objects)
        else:
            self.covered_objects &= set(condition.covered_objects)

        if condition in self.conditions:
            raise ValueError(f"Condition {condition} already exists in conjunction")

        self.conditions.append(condition)
        self._compile_rule(with_lambda=False)
        return self

    def remove_redundant(self, objects_b: set | list) -> None:
        """Removes redundant conditions from the conjunction based on a given set or list of objects."""
        i = 0
        while i < len(self.conditions):
            if len(self.conditions) == 1:
                break
            if Conjunction(self.conditions[:i] + self.conditions[i + 1 :]).covered_objects.issubset(objects_b):
                self.conditions.pop(i)
            else:
                i += 1
        self._compile_rule()


class Rule:
    def __init__(self, condition: Conjunction, decision_class: numeric, at_most: bool) -> None:
        self.conditions = condition
        self.decision = decision_class
        self.operator = Operator.LE if at_most else Operator.GE

        self.support = ...
        self.strength = ...
        self.coverage_factor = ...
        self.confidence = ...
        self.epsilon = ...

        self.conditions_str = f"if {condition} then {self.operator.value} {decision_class}"

    def do_match(self, **kwargs) -> bool:
        return self.conditions._compiled_conjunction(**kwargs)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Rule):
            return NotImplemented
        return (
            self.operator == __value.operator
            and self.decision == __value.decision
            and self.conditions == __value.conditions
        )

    def __repr__(self) -> str:
        return self.conditions_str
