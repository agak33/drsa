from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd

from .criterion import Criterion, DecisionCriterion
from .rule import Condition, Conjunction, Operator, Rule, RulesType
from .types import ApproximationType, ClassApproximation, Dominance, DominanceType, numeric


class Algorithm(Enum):
    DOMLEM = "domlem"


class AlternativesSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df

        self.rules_certain: list[Rule] = []
        self.rules_possible: list[Rule] = []
        self.rules_approximate: list[Rule] = []

        self.positive_cones: dict[str, Dominance] = {}
        self.negative_cones: dict[str, Dominance] = {}

        self.at_least_approximations: dict[numeric, ClassApproximation] = {}
        self.at_most_approximations: dict[numeric, ClassApproximation] = {}

        self.decision_attributes: list = []
        self.cost_attributes: list = []
        self.gain_attributes: list = []
        for column in self.data.columns:
            if isinstance(column, Criterion):
                if column.is_cost:
                    self.cost_attributes = [column] if self.cost_attributes is None else [*self.cost_attributes, column]
                else:
                    self.gain_attributes = [column] if self.gain_attributes is None else [*self.gain_attributes, column]
            elif isinstance(column, DecisionCriterion):
                self.decision_attributes = (
                    [column] if self.decision_attributes is None else [*self.decision_attributes, column]
                )

        if not self.decision_attributes:
            raise ValueError("No decision attributes found")

        if not self.cost_attributes and not self.gain_attributes:
            raise ValueError("No attributes found")

    def _check_domination(self, alternative_a: pd.Series) -> tuple[Dominance, Dominance]:
        """Check, which alternatives are dominated by the given one."""

        domination = pd.DataFrame(
            np.concatenate(
                [
                    np.where(
                        alternative_a[self.cost_attributes] < self.data[self.cost_attributes],
                        1,
                        np.where(
                            alternative_a[self.cost_attributes] > self.data[self.cost_attributes],
                            -1,
                            0,
                        ),
                    ),
                    np.where(
                        alternative_a[self.gain_attributes] > self.data[self.gain_attributes],
                        1,
                        np.where(
                            alternative_a[self.gain_attributes] < self.data[self.gain_attributes],
                            -1,
                            0,
                        ),
                    ),
                    self.data[self.decision_attributes].to_numpy(),
                ],
                axis=1,
            ),
            index=self.data.index,
            columns=self.cost_attributes + self.gain_attributes + self.decision_attributes,
        )

        # alternative_a is dominating these
        dominating = domination[(domination[self.cost_attributes + self.gain_attributes] != -1).all(axis=1)]

        # alternative_a is dominated by these
        dominated = domination[(domination[self.cost_attributes + self.gain_attributes] != 1).all(axis=1)]

        return (
            # alternative_a is dominating these
            Dominance(
                alternative_a.name,
                dominating.index.values,
                DominanceType.DOMINATING,
                alternative_a[self.decision_attributes][0],
                dominating[self.decision_attributes].min()[0],
                dominating[self.decision_attributes].max()[0],
            ),
            # alternative_a is dominated by these
            Dominance(
                alternative_a.name,
                dominated.index.values,
                DominanceType.DOMINATED,
                alternative_a[self.decision_attributes][0],
                dominated[self.decision_attributes].min()[0],
                dominated[self.decision_attributes].max()[0],
            ),
        )

    def dominance_cones(
        self,
        alternatives_subset: Iterable[str] = [],
    ) -> None:
        """Calculates dominance cones for given alternatives (negative and positive).

        :param alternatives_subset: if non-empty, cones won't be calculated
        for all alternatives, defaults to ``[]``

        After calling this method, the results are stored in `negative_cones`
        and `positive_cones` attributes.
        """
        if not alternatives_subset:
            alternatives_subset = self.data.index.values

        self.negative_cones = {}
        self.positive_cones = {}
        for alternative_a in alternatives_subset:
            dominates, is_dominated = self._check_domination(self.data.loc[alternative_a])

            self.negative_cones[alternative_a] = dominates
            self.positive_cones[alternative_a] = is_dominated

    def class_unions(self) -> None:
        if not self.positive_cones or not self.negative_cones:
            self.dominance_cones()

        class_set = sorted(self.data[self.decision_attributes[0]].unique())

        self.at_least_approximations = {}
        self.at_most_approximations = {}

        # at least approximation
        for class_value in class_set[1:]:
            # lower approximation
            lower_approximation = [
                obj.object_name for obj in self.positive_cones.values() if obj.min_class >= class_value
            ]

            # upper approximation
            upper_approximation = [
                obj.object_name
                for obj, obj_neg in zip(self.positive_cones.values(), self.negative_cones.values())
                if obj.object_class >= class_value or obj_neg.max_class >= class_value
            ]

            objects_that_belong = [
                obj.object_name for obj in self.positive_cones.values() if obj.object_class >= class_value
            ]

            self.at_least_approximations[class_value] = ClassApproximation(
                class_value,
                ApproximationType.AT_LEAST,
                lower_approximation,
                upper_approximation,
                objects_that_belong,
            )

        # at most approximation
        for class_value in class_set[:-1]:
            # lower approximation
            lower_approximation = [
                obj.object_name for obj in self.negative_cones.values() if obj.max_class <= class_value
            ]

            # upper approximation
            upper_approximation = [
                obj.object_name
                for obj, obj_pos in zip(self.negative_cones.values(), self.positive_cones.values())
                if obj.object_class <= class_value or obj_pos.min_class <= class_value
            ]

            # objects that belong to the approximation
            objects_that_belong = [
                obj.object_name for obj in self.negative_cones.values() if obj.object_class <= class_value
            ]

            self.at_most_approximations[class_value] = ClassApproximation(
                class_value,
                ApproximationType.AT_MOST,
                lower_approximation,
                upper_approximation,
                objects_that_belong,
            )

    def reductors(self) -> None: ...

    def core(self) -> None: ...

    def _get_candidate_conditions(
        self, not_covered_objects: list | set, rules_at_most: bool = False
    ) -> list[Condition]:
        cost_operator, gain_operator = (Operator.GE, Operator.LE) if rules_at_most else (Operator.LE, Operator.GE)
        data_subset = self.data.loc[not_covered_objects]

        return [
            Condition(
                column,
                cost_operator,
                value,
                self.data[eval(f"self.data[column] {cost_operator.value} {value}")].index.values,
                not_covered_objects,
            )
            for column in self.cost_attributes
            for value in data_subset[column].unique()
        ] + [
            Condition(
                column,
                gain_operator,
                value,
                self.data[eval(f"self.data[column] {gain_operator.value} {value}")].index.values,
                not_covered_objects,
            )
            for column in self.gain_attributes
            for value in data_subset[column].unique()
        ]

    def _domlem(self, type: RulesType, debug=False) -> list[Rule]:
        if not self.at_least_approximations or not self.at_most_approximations:
            self.class_unions()

        rules_set = []
        # at most rules, then at least rules
        for at_most_rules, class_approximation in zip(
            (False, True), (dict(reversed(list(self.at_least_approximations.items()))), self.at_most_approximations)
        ):
            for class_value, approximation in class_approximation.items():
                # G := B; (a set of objects not covered by the conjunction T∈T
                all_objects = (
                    approximation.lower_approximation
                    if type == RulesType.CERTAIN
                    else approximation.upper_approximation
                )
                objects_to_cover: set | list = all_objects.copy()
                if debug:
                    print(
                        f"- Looking for rules for {'at most' if at_most_rules else 'at least'} {class_value}."
                        f"All objects: {objects_to_cover}"
                    )

                # T := ∅; (a set of conjunctions holding the current coverage of set B\G)
                set_of_conjunctions: list[Conjunction] = []

                # while G ≠ ∅ (there are still some objects to be covered)
                while objects_to_cover:
                    # T := ∅; (conjunction of conditions; a candidate for condition part of the rule)
                    conjunction = Conjunction()

                    if debug:
                        print(
                            f"-- Initiated new empty conjunction {conjunction}. "
                            f"Remaining objects to cover: {objects_to_cover}"
                        )

                    # T(G) := {t = (q ≻ rq) : ∃x∈G: xq = rq} (a set of candidate conditions for not covered objects)
                    candidates = self._get_candidate_conditions(objects_to_cover, rules_at_most=at_most_rules)
                    if debug:
                        print(f"-- Initiates candidates: {candidates}")

                    # while T = ∅ or [T] ⊄ B (the conjunction needs to cover only objects from B)
                    while not conjunction.conditions or not set(conjunction.covered_objects).issubset(all_objects):
                        if debug:
                            print(
                                f"--- Looking for a new condition; {conjunction.conditions},"
                                f"{set(conjunction.covered_objects)}"
                            )
                        # select t ∈ T(G) such that |{t} ∩ G|/|{t}| is maximum; if a tie occurs, select a pair such
                        # that |{t} ∩ G| is maximum; if another tie occurs, select the first pair;
                        t = max(candidates, key=lambda x: x.cover_factor)

                        if debug:
                            print(f"--- Chosen candidate: {t}")

                        # T := T ∪ t; (add condition t to the conjunction T}
                        conjunction = conjunction + t
                        if debug:
                            print(f"--- Updated conjunction: {conjunction.conditions}")

                        # G:= G ∩ [t]; (reduce the set of objects generating new elementary conditions)
                        objects_to_cover = list(set(objects_to_cover) & set(t.covered_objects))
                        if debug:
                            print(f"--- Updated objects to cover: {objects_to_cover}")

                        # T(G) := {w = (q ≻ rq) : ∃x∈G: xq = rq}; (update a set of candidate conditions)
                        candidates = self._get_candidate_conditions(objects_to_cover, rules_at_most=at_most_rules)
                        if debug:
                            print(f"--- Updated candidates: {candidates}")

                        # T(G) := T(G) – T; (eliminate the already used conditions from the list)
                        for c in candidates[:]:
                            if c in conjunction.conditions:
                                if debug:
                                    print(
                                        f"---- Removing {c} from candidates, because it's already in "
                                        f"the conjunction {conjunction}"
                                    )
                                candidates.remove(c)

                    # for each elementary condition t ∈ T do
                    # if [T – {t}] ⊆ B then T := T – t; (eliminate redundant conditions)
                    conjunction.remove_redundant(all_objects)
                    if debug:
                        print(f"-- Final conjunction: {conjunction}")

                    # T := T ∪ T; (add the minimal conjunction to the set of conjunctions)
                    set_of_conjunctions.append(conjunction)
                    if debug:
                        print(f"-- Updated set of conjunctions: {set_of_conjunctions}")

                    # G := B - ∪T∈T [T]; (update the set of objects to be covered)
                    objects_to_cover = set(all_objects)
                    for conj in set_of_conjunctions:
                        objects_to_cover -= conj.covered_objects
                    objects_to_cover = list(objects_to_cover)
                    if debug:
                        print(f"-- Updated objects to cover: {objects_to_cover}")

                if debug:
                    print("-----")
                # for each conjunction T∈T do
                # if ∪K∈T\T [K] = B then T := T – T; (eliminate redundant conjunctions)
                i = 0
                while i < len(set_of_conjunctions):
                    if len(set_of_conjunctions) > 1 and set.union(
                        *[set(k.covered_objects) for k in set_of_conjunctions[:i] + set_of_conjunctions[i + 1 :]]
                    ) == set(approximation.lower_approximation):
                        set_of_conjunctions.pop(i)
                    else:
                        # create rules R based on all conjunctions T∈T;
                        rules_set.append(
                            Rule(condition=set_of_conjunctions[i], decision_class=class_value, at_most=at_most_rules)
                        )
                        i += 1

        return rules_set

    def rules(self, algorithm: Algorithm = Algorithm.DOMLEM, rules_type: RulesType = RulesType.CERTAIN) -> list[Rule]:
        if algorithm == Algorithm.DOMLEM:
            rules = self._domlem(rules_type)
            if rules_type == RulesType.CERTAIN:
                self.rules_certain = rules
            elif rules_type == RulesType.POSSIBLE:
                self.rules_possible = rules
            elif rules_type == RulesType.APPROXIMATE:
                self.rules_approximate = rules
            return rules

        raise ValueError(f"Algorithm {algorithm} is not supported.")
