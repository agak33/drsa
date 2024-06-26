import math
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .criterion import Criterion, DecisionCriterion
from .rule import Condition, Conjunction, Operator, Rule, RulesType
from .types import ApproximationType, ClassApproximation, Dominance, DominanceType, numeric


class Algorithm(Enum):
    DOMLEM = "domlem"
    VC_DOMLEM = "vc-domlem"


class Metric(Enum):
    GAIN_TYPE = "gain"
    COST_TYPE = "cost"


class AlternativesSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df

        self.positive_cones: dict[str, Dominance] = {}
        self.negative_cones: dict[str, Dominance] = {}

        self.lower_approximations_at_least: dict[numeric, set] = {}
        self.lower_approximations_at_most: dict[numeric, set] = {}

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

        assert len(self.decision_attributes) == 1, "Only one decision attribute is supported"
        self.decision_attribute = self.decision_attributes[0]

        self.upward_class_unions = self._calculate_class_unions(upward=True)
        self.downward_class_unions = self._calculate_class_unions(upward=False)

        self.epsilons_upward: dict[Any, dict[numeric, float]] = {}
        self.epsilons_downward: dict[Any, dict[numeric, float]] = {}

        self.mu_upward: dict[Any, dict[numeric, float]] = {}
        self.mu_downward: dict[Any, dict[numeric, float]] = {}

        self.dominance_cones()
        self._calculate_metrics()

        # for each column, save all possible values (sorted)
        self.values_in_attributes: dict[Criterion, list[numeric]] = {
            column: sorted(self.data[column].unique()) for column in self.cost_attributes + self.gain_attributes
        }

        # for each class, save all cases
        self.objects_in_classes: dict[numeric, set] = {
            class_value: set(self.data[self.data[self.decision_attribute] == class_value].index)
            for class_value in self.data[self.decision_attribute].unique()
        }

        self.all_objects = set(self.data.index.values)

    def _calculate_class_unions(self, upward: bool = False) -> dict[numeric, set]:
        """Caluclate class unions for given alternatives.
        Upward union: Cl_t>=
        Downward union: Cl_t<=
        """
        class_set = sorted(self.data[self.decision_attribute].unique())

        class_unions = {}
        for class_value in class_set:
            class_unions[class_value] = set(
                self.data[self.data[self.decision_attribute] >= class_value].index
                if upward
                else self.data[self.data[self.decision_attribute] <= class_value].index
            )

        return class_unions

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

    def _get_gain_metric_value(
        self,
        objects_in_cone: set[numeric],
        class_value: numeric,
        upward: bool = False,
    ) -> float:
        if upward:
            return len(objects_in_cone & self.upward_class_unions[class_value]) / len(objects_in_cone)
        return len(objects_in_cone & self.downward_class_unions[class_value]) / len(objects_in_cone)

    def _get_cost_metric_value(
        self,
        objects_in_cone: set[numeric],
        class_value: numeric,
        upward: bool = False,
    ) -> float:
        class_union = (
            self.downward_class_unions[class_value - 1] if upward else self.upward_class_unions[class_value + 1]
        )
        return len(objects_in_cone & class_union) / len(class_union)

    def _calculate_metrics(self) -> None:
        """Calculate epsilon and mu metrics for each object."""
        class_set = sorted(self.data[self.decision_attributes[0]].unique())

        for alternative_a in self.data.index.values:
            epsilons_upward = {}
            epsilons_downward = {}
            mu_upward = {}
            mu_downward = {}

            # at least (upward)
            for class_value in class_set[1:]:
                epsilons_upward[class_value] = self._get_cost_metric_value(
                    set(self.positive_cones[alternative_a].objects), class_value, upward=True
                )
                mu_upward[class_value] = self._get_gain_metric_value(
                    set(self.positive_cones[alternative_a].objects), class_value, upward=True
                )

            # at most (downward)
            for class_value in class_set[:-1]:
                epsilons_downward[class_value] = self._get_cost_metric_value(
                    set(self.negative_cones[alternative_a].objects), class_value, upward=False
                )
                mu_downward[class_value] = self._get_gain_metric_value(
                    set(self.negative_cones[alternative_a].objects), class_value, upward=False
                )

            self.epsilons_upward[alternative_a] = {
                class_value: max([val for key, val in epsilons_upward.items() if key <= class_value])
                for class_value in class_set[1:]
            }
            self.epsilons_downward[alternative_a] = {
                class_value: max([val for key, val in epsilons_downward.items() if key >= class_value])
                for class_value in class_set[:-1]
            }
            self.mu_upward[alternative_a] = mu_upward
            self.mu_downward[alternative_a] = mu_downward

    def _calculate_lower_approximations(
        self,
        threshold: float = 0.0,
        metric_type: Metric = Metric.GAIN_TYPE,
    ) -> None:
        if not self.positive_cones or not self.negative_cones:
            self.dominance_cones()

        class_set = sorted(self.data[self.decision_attributes[0]].unique())

        # at least (upward) approximation
        for class_value in class_set[1:]:
            if not threshold:
                self.lower_approximations_at_least[class_value] = set(
                    obj.object_name for obj in self.positive_cones.values() if obj.min_class >= class_value
                )
            elif metric_type == Metric.GAIN_TYPE:
                self.lower_approximations_at_least[class_value] = set(
                    obj.object_name
                    for obj in self.positive_cones.values()
                    if self.mu_upward[obj.object_name][class_value] >= threshold
                    and obj.object_name in self.upward_class_unions[class_value]
                )
            elif metric_type == Metric.COST_TYPE:
                self.lower_approximations_at_least[class_value] = set(
                    obj.object_name
                    for obj in self.positive_cones.values()
                    if self.epsilons_upward[obj.object_name][class_value] <= threshold
                    and obj.object_name in self.upward_class_unions[class_value]
                )

        # at most (downward) approximation
        for class_value in class_set[:-1]:
            if not threshold:
                self.lower_approximations_at_most[class_value] = set(
                    obj.object_name for obj in self.negative_cones.values() if obj.max_class <= class_value
                )
            elif metric_type == Metric.GAIN_TYPE:
                self.lower_approximations_at_most[class_value] = set(
                    obj.object_name
                    for obj in self.negative_cones.values()
                    if self.mu_downward[obj.object_name][class_value] >= threshold
                    and obj.object_name in self.downward_class_unions[class_value]
                )
            elif metric_type == Metric.COST_TYPE:
                self.lower_approximations_at_most[class_value] = set(
                    obj.object_name
                    for obj in self.negative_cones.values()
                    if self.epsilons_downward[obj.object_name][class_value] <= threshold
                    and obj.object_name in self.downward_class_unions[class_value]
                )

    def class_approximations(self, threshold: float = 0.0, metric_type: Metric = Metric.GAIN_TYPE) -> None:
        if not self.positive_cones or not self.negative_cones:
            self.dominance_cones()

        self._calculate_lower_approximations(threshold=threshold, metric_type=metric_type)
        class_set = sorted(self.data[self.decision_attributes[0]].unique())

        self.at_least_approximations = {}
        self.at_most_approximations = {}

        # at least approximation
        for class_value in class_set[1:]:
            # upper approximation
            upper_approximation = self.all_objects - self.lower_approximations_at_most.get(class_value - 1, set())

            positive_region = set.union(
                *[set(self.positive_cones[obj].objects) for obj in self.lower_approximations_at_least[class_value]]
            )

            self.at_least_approximations[class_value] = ClassApproximation(
                class_value=class_value,
                class_union=self.upward_class_unions[class_value],
                approximation_type=ApproximationType.AT_LEAST,
                lower_approximation=self.lower_approximations_at_least[class_value],
                upper_approximation=upper_approximation,
                positive_region=positive_region,
            )

        # at most approximation
        for class_value in class_set[:-1]:
            # upper approximation
            upper_approximation = self.all_objects - self.lower_approximations_at_least.get(class_value + 1, set())

            positive_region = set.union(
                *[set(self.negative_cones[obj].objects) for obj in self.lower_approximations_at_most[class_value]]
            )

            self.at_most_approximations[class_value] = ClassApproximation(
                class_value=class_value,
                class_union=self.downward_class_unions[class_value],
                approximation_type=ApproximationType.AT_MOST,
                lower_approximation=self.lower_approximations_at_most[class_value],
                upper_approximation=upper_approximation,
                positive_region=positive_region,
            )

    def reductors(self) -> None: ...

    def core(self) -> None: ...

    def _get_candidate_conditions(
        self, not_covered_objects: list | set, rules_at_most: bool = False
    ) -> list[Condition]:
        cost_operator, gain_operator = (Operator.GE, Operator.LE) if rules_at_most else (Operator.LE, Operator.GE)
        data_subset = self.data.loc[list(not_covered_objects)]

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

    def _domlem(self, type: RulesType) -> list[Rule]:
        if not self.at_least_approximations or not self.at_most_approximations:
            self.class_approximations()

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

                # T := ∅; (a set of conjunctions holding the current coverage of set B\G)
                set_of_conjunctions: list[Conjunction] = []

                # while G ≠ ∅ (there are still some objects to be covered)
                while objects_to_cover:
                    # T := ∅; (conjunction of conditions; a candidate for condition part of the rule)
                    conjunction = Conjunction()

                    # T(G) := {t = (q ≻ rq) : ∃x∈G: xq = rq} (a set of candidate conditions for not covered objects)
                    candidates = self._get_candidate_conditions(objects_to_cover, rules_at_most=at_most_rules)

                    # while T = ∅ or [T] ⊄ B (the conjunction needs to cover only objects from B)
                    while not conjunction.conditions or not set(conjunction.covered_objects).issubset(all_objects):
                        # select t ∈ T(G) such that |{t} ∩ G|/|{t}| is maximum; if a tie occurs, select a pair such
                        # that |{t} ∩ G| is maximum; if another tie occurs, select the first pair;
                        t = max(candidates, key=lambda x: x.cover_factor)

                        # T := T ∪ t; (add condition t to the conjunction T}
                        conjunction = conjunction + t

                        # G:= G ∩ [t]; (reduce the set of objects generating new elementary conditions)
                        objects_to_cover = list(set(objects_to_cover) & set(t.covered_objects))

                        # T(G) := {w = (q ≻ rq) : ∃x∈G: xq = rq}; (update a set of candidate conditions)
                        candidates = self._get_candidate_conditions(objects_to_cover, rules_at_most=at_most_rules)

                        # T(G) := T(G) – T; (eliminate the already used conditions from the list)
                        for c in candidates[:]:
                            if c in conjunction.conditions:
                                candidates.remove(c)

                    # for each elementary condition t ∈ T do
                    # if [T – {t}] ⊆ B then T := T – t; (eliminate redundant conditions)
                    conjunction.remove_redundant(all_objects)

                    # T := T ∪ T; (add the minimal conjunction to the set of conjunctions)
                    set_of_conjunctions.append(conjunction)

                    # G := B - ∪T∈T [T]; (update the set of objects to be covered)
                    objects_to_cover = set(all_objects)
                    for conj in set_of_conjunctions:
                        objects_to_cover -= conj.covered_objects
                    objects_to_cover = list(objects_to_cover)

                # for each conjunction T∈T do
                # if ∪K∈T\T [K] = B then T := T – T; (eliminate redundant conjunctions)
                i = 0
                while i < len(set_of_conjunctions):
                    if len(set_of_conjunctions) > 1 and set.union(
                        *[set(k.covered_objects) for k in set_of_conjunctions[:i] + set_of_conjunctions[i + 1 :]]
                    ) == set(all_objects):
                        set_of_conjunctions.pop(i)
                    else:
                        # create rules R based on all conjunctions T∈T;
                        rules_set.append(
                            Rule(
                                condition=set_of_conjunctions[i],
                                decision_class=class_value,
                                at_most=at_most_rules,
                                objects_in_each_class=self.objects_in_classes,
                                all_objects=self.all_objects,
                                class_union=set(approximation.class_union),
                            )
                        )
                        i += 1

        return rules_set

    def _allowed_objects(
        self,
        approximation: ClassApproximation,
        # threshold: float, # ????
        covering_option: int,
        upward: bool = False,
    ) -> set:
        if covering_option == 1:
            return set(approximation.positive_region)
        elif covering_option == 2:
            return set(approximation.positive_region).union(set(approximation.boundary))
        elif covering_option == 3:
            return self.all_objects
        raise ValueError(f"Covering option {covering_option} is not supported.")

    def _rule_metric_cost(self, covered_objects: set, lower_approximation: set) -> float:
        if not covered_objects:
            return np.inf
        complement = self.all_objects - lower_approximation
        return len(covered_objects & complement) / len(complement)

    def _bayesian_confirmation_measure(self, rule: Rule) -> float:
        """Calculate Bayesian confirmation measure for a given rule (P(h|e) / P(h))."""
        # 1. P(h|e)
        # probability, that object belong to decision part of the rule
        # provided that it meets the conditions of the rule
        p_he = len(rule.conditions.covered_objects) / len(
            self.data[eval(f"self.data[self.decision_attribute] {rule.operator.value} {rule.decision}")].index.values
        )

        # 2. P(h)
        # probability, that any random object belong to decision part of the rule
        p_h = len(
            self.downward_class_unions[rule.decision]
            if rule.operator == Operator.LE
            else self.upward_class_unions[rule.decision]
        ) / len(self.all_objects)

        return p_he / p_h if p_h else 0.0

    def _generalize_condition(
        self,
        elementary_condition: Condition,
        rest_conditions: list[Condition],
        threshold: float,
        lower_approximation: set,
        allowed_objects: set,
    ) -> Condition:
        # index of value in attribute (the list is sorted)
        value_index = self.values_in_attributes[elementary_condition.field].index(elementary_condition.value)

        while (value_index > 0 and elementary_condition.operator == Operator.GE) or (
            value_index < len(self.values_in_attributes[elementary_condition.field]) - 1
            and elementary_condition.operator == Operator.LE
        ):
            # attr <= value -> try to make value higher, lower otherwise
            value_index += 1 if elementary_condition.operator == Operator.LE else -1
            new_value = self.values_in_attributes[elementary_condition.field][value_index]

            new_condition = Condition(
                field=elementary_condition.field,
                operator=elementary_condition.operator,
                value=new_value,
                covered_objects=self.data[
                    eval(f"self.data[elementary_condition.field] {elementary_condition.operator.value} {new_value}")
                ].index.values,
            )
            if self._rule_metric_cost(
                covered_objects=Conjunction(conditions=rest_conditions + [new_condition]).covered_objects,
                lower_approximation=lower_approximation,
            ) > threshold or not Conjunction(conditions=rest_conditions + [new_condition]).covered_objects.issubset(
                allowed_objects
            ):
                return elementary_condition

            elementary_condition = new_condition

        return elementary_condition

    def _generalize_rule(
        self,
        conjunction: Conjunction,
        threshold: float,
        lower_approximation: set,
        allowed_objects: set,
    ) -> Conjunction:
        # iterate over ec in r_x (from oldest to newest)
        i = 0
        while i < len(conjunction.conditions):
            # try to make ec more general, as long as threshold is not exceeded
            conjunction.conditions[i] = self._generalize_condition(
                elementary_condition=conjunction.conditions[i],
                rest_conditions=conjunction.conditions[:i] + conjunction.conditions[i + 1 :],
                threshold=threshold,
                lower_approximation=lower_approximation,
                allowed_objects=allowed_objects,
            )
            i += 1

        conjunction._compile_rule()
        return conjunction

    def _vc_domlem(
        self,
        covering_option: int,
        threshold: float,
        extended: bool = False,
        confidence_threshold: float | None = None,
        coverage_threshold: float | None = None,
    ) -> list[Rule]:
        if not self.at_least_approximations or not self.at_most_approximations:
            self.class_approximations(threshold=threshold, metric_type=Metric.COST_TYPE)

        rules_set = []
        # at most rules, then at least rules
        for at_most_rules, class_approximation in zip(
            (False, True), (dict(reversed(list(self.at_least_approximations.items()))), self.at_most_approximations)
        ):
            for class_value, approximation in class_approximation.items():
                # allowed objects
                allowed_objects = self._allowed_objects(
                    approximation=approximation,
                    covering_option=covering_option,
                )

                # vc-sequential covering
                objects_to_cover: set | list = approximation.lower_approximation.copy()
                set_of_conjunctions: list[Conjunction] = []

                while objects_to_cover:
                    conjunction = Conjunction()
                    candidates = self._get_candidate_conditions(list(objects_to_cover), rules_at_most=at_most_rules)

                    while self._rule_metric_cost(
                        covered_objects=conjunction.covered_objects,
                        lower_approximation=set(approximation.lower_approximation),
                    ) > threshold or not conjunction.covered_objects.issubset(allowed_objects):
                        # best elementary condition
                        t = max(
                            candidates,
                            key=lambda x: (
                                -self._rule_metric_cost(
                                    conjunction.covered_objects & set(x.covered_objects),
                                    allowed_objects,
                                ),
                                len(set(x.covered_objects) & set(approximation.lower_approximation)),
                            ),
                        )

                        conjunction = conjunction + t

                        candidates = self._get_candidate_conditions(
                            list(set(objects_to_cover) & conjunction.covered_objects), rules_at_most=at_most_rules
                        )
                        for c in candidates[:]:
                            if c in conjunction.conditions:
                                candidates.remove(c)

                    # remove redundant ec from conjunction
                    # TODO check thresholds
                    conjunction.remove_redundant(allowed_objects)

                    if extended:
                        # generalize elementary conditions
                        print(f"Before generalization: {conjunction}")
                        conjunction = self._generalize_rule(
                            conjunction=conjunction,
                            threshold=threshold,
                            lower_approximation=set(approximation.lower_approximation),
                            allowed_objects=allowed_objects,
                        )
                        print(f"After generalization : {conjunction}")

                    # add rule
                    set_of_conjunctions.append(conjunction)

                    objects_to_cover = set(objects_to_cover) - conjunction.covered_objects

                # remove redundant rules from set_of_conjunctions
                # add set_of_conjunctions to rules_set
                # remove non-minimal rules from rules_set
                i = 0
                while i < len(set_of_conjunctions):
                    if len(set_of_conjunctions) > 1 and set.union(
                        *[set(k.covered_objects) for k in set_of_conjunctions[:i] + set_of_conjunctions[i + 1 :]]
                    ) == set(allowed_objects):
                        set_of_conjunctions.pop(i)
                    else:
                        # create rules R based on all conjunctions T∈T;
                        rules_set.append(
                            Rule(
                                condition=set_of_conjunctions[i],
                                decision_class=class_value,
                                at_most=at_most_rules,
                                objects_in_each_class=self.objects_in_classes,
                                all_objects=self.all_objects,
                                class_union=set(approximation.class_union),
                            )
                        )
                        i += 1

        if extended:
            for rule in rules_set:
                if math.isclose(self._bayesian_confirmation_measure(rule), 0.0):
                    print(f"Removing rule: {rule}")
                    rules_set.remove(rule)

                elif confidence_threshold and rule.confidence < confidence_threshold:
                    print(f"Removing rule: {rule}")
                    rules_set.remove(rule)

                elif coverage_threshold and rule.coverage_factor < coverage_threshold:
                    print(f"Removing rule: {rule}")
                    rules_set.remove(rule)
        return rules_set

    def rules(
        self,
        algorithm: Algorithm = Algorithm.DOMLEM,
        /,
        rules_type: RulesType = RulesType.CERTAIN,
        vc_threshold: float = 0.0,
        vc_covering_option: int = 1,
        vc_extended: bool = False,
        vc_confidence_threshold: float | None = None,
        vc_coverage_threshold: float | None = None,
    ) -> list[Rule]:
        if algorithm == Algorithm.DOMLEM:
            return self._domlem(rules_type)
        elif algorithm == Algorithm.VC_DOMLEM:
            return self._vc_domlem(
                threshold=vc_threshold,
                covering_option=vc_covering_option,
                extended=vc_extended,
                confidence_threshold=vc_confidence_threshold,
                coverage_threshold=vc_coverage_threshold,
            )

        raise ValueError(f"Algorithm {algorithm} is not supported.")
