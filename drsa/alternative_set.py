from typing import Iterable

import numpy as np
import pandas as pd

from .criterion import Criterion, DecisionCriterion
from .types import ApproximationType, ClassApproximation, Dominance, DominanceType, numeric


class AlternativesSet:
    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df

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
