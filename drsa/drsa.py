"""This package implements some of the DRSA methods. :)
"""
import math
from typing import Iterable

import pandas as pd

from .types import RelationType, numeric


class Criterion:
    def __init__(
        self,
        name: str,
        is_cost: bool = False,
        is_decision_attr: bool = False,
    ) -> None:
        self.name = name

        self.is_cost = None if is_decision_attr else is_cost
        self.is_decision_attr = is_decision_attr

    def __repr__(self) -> str:
        return (
            f"{self.name}; "
            f"{'decision' if self.is_decision_attr else 'cost' if self.is_cost else 'gain'} attr"
        )


class Alternative(pd.Series):
    def __init__(
        self,
        name: str,
        data: Iterable[numeric],
        index: list[Criterion],
        *args,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__(data, name=name, index=index, *args, **kwargs)


class AlternativesSet(pd.DataFrame):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.decision_attr = None
        for attr in self.columns.values:
            if attr.is_decision_attr:
                if self.decision_attr is not None:
                    raise ValueError("The set has more than one decision attribute.")
                self.decision_attr = attr

        if self.decision_attr is None:
            raise ValueError("The set has no decision attribute.")

    def _check_domination_for_criterion(
        self, value_a: numeric, value_b: numeric, is_cost_attr: bool | None
    ) -> int:
        """_summary_

        :param value_a: Criterion's value of one alternative
        :param value_b: Criterion's value (the same as the `value_a`) of the second alternative
        :param is_cost_attr: Defines if the criterion is cost type
        (if it's a decision criterion, this argument should be a ``None``)

        :return:
            * ``0`` - if `a` is indifferent to `b` on the given
              criterion / the criterion is decision one
            * ``1`` - if `a` dominates `b` on the given criterion
            * ``-1`` - otherwise (`b` dominates `a`)
        """
        if is_cost_attr is None or value_a == value_b:
            return 0

        if is_cost_attr:
            return -1 if value_a > value_b else 1

        return 1 if value_a > value_b else -1

    def _check_domination_pair(
        self, alternative_a: str, alternative_b: str, criteria: list[Criterion] = []
    ) -> RelationType:
        """Checks domination type for one pair of alternatives.

        :param alternative_a: an `Alternative` objects
        :param alternative_b: an `Alternative` objects
        :param criteria: if set to non-empty list, cones will be calculated only
        for given subset of criteria, defaults to ``[]``

        :return: one of the `RelationType` values
        """
        alt_a: pd.Series = self.loc[alternative_a]
        alt_b: pd.Series = self.loc[alternative_b]
        criteria = list(alt_a.keys()) if not criteria else criteria

        indifferent_count = 0
        domination_count = 0
        decision_count = 0

        for value_a, value_b, criterion in zip(alt_a[criteria], alt_b[criteria], criteria):
            if not criterion.is_decision_attr:
                if value_a == value_b:
                    indifferent_count += 1
                else:
                    domination_count += self._check_domination_for_criterion(
                        value_a, value_b, criterion.is_cost
                    )
            else:
                decision_count += 1

        if indifferent_count == len(criteria) - decision_count:
            return RelationType.INDIFFERENT

        if domination_count == len(criteria) - decision_count - indifferent_count:
            return RelationType.DOMINATING

        if -domination_count == len(criteria) - decision_count - indifferent_count:
            return RelationType.DOMINATED

        return RelationType.INCOMPARABLE

    def dominance_cones(
        self,
        negative: bool = False,
        alternatives_subset: Iterable[str] = [],
        criteria: list[Criterion] = [],
    ) -> pd.Series:
        """Returns dominance cones, for all alternatives by default.

        :param negative: if ``True``, calculates negative cones,
        positive ones otherwise, defaults to ``False``
        :param alternatives_subset: if non-empty, cones won't be calculated
        for all alternatives, defaults to ``[]``
        :param criteria: if set to non-empty list, cones will be calculated only
        for given subset of criteria, defaults to ``[]``

        :return: a `pandas.Series` object, indexed with alternatives names,
        valued with sets of alternatives (representing cones)
        """
        if not alternatives_subset:
            alternatives_subset = self.index.values

        result = pd.Series(
            [set() for _ in range(len(list(alternatives_subset)))], index=alternatives_subset
        )
        for alternative_a in alternatives_subset:
            for alternative_b in self.index.values:
                relation = self._check_domination_pair(alternative_a, alternative_b, criteria)

                if negative and relation in [RelationType.INDIFFERENT, RelationType.DOMINATING]:
                    result[alternative_a].add(alternative_b)

                elif not negative and relation in [
                    RelationType.INDIFFERENT,
                    RelationType.DOMINATED,
                ]:
                    result[alternative_a].add(alternative_b)

        return result

    def _get_cones_min_max_class(self, dominance_cones: pd.Series) -> pd.Series:
        """Transforms dominance cones into lists with information
        about minimal and maximal class.

        :param dominance_cones: a `pandas.Series` object containing dominance cones

        :return: a `pandas.Series` object; the index stays the same as in `dominance_cones`,
        and values are 2-element lists in the form of `[min_class, max_class]`
        """
        result = pd.Series([])
        for x in dominance_cones.index.values:
            classes = [self[self.decision_attr][el] for el in dominance_cones[x]]
            result[x] = [min(classes), max(classes)]
        return result

    def class_approximation(
        self,
        at_most: bool = False,
        criteria: list[Criterion] = [],
    ) -> tuple[pd.Series, pd.Series]:
        """Calculates lower and upper class approximations.

        :param at_most: If ``True``, the method would return 'at most' approximations,
        'at least' otherwise, defaults to ``False``

        :return: a `tuple` object with two `pandas.Series` - the first one with
        lower approximations, second with upper one. Series has names with information
        about its type :)
        """
        cones = (
            self.dominance_cones(negative=True, criteria=criteria)
            if at_most
            else self.dominance_cones(criteria=criteria)
        )
        class_set = sorted(self[self.decision_attr].unique())

        result_lower = pd.Series(
            {cls: set() for cls in class_set},
            name=f"lower approximation, {'at most' if at_most else 'at least'}",
        )
        result_upper = pd.Series(
            {cls: set() for cls in class_set},
            name=f"upper approximation, {'at most' if at_most else 'at least'}",
        )

        cone_min_max = self._get_cones_min_max_class(cones)

        for cls in class_set:
            # lower
            for x, [min, max] in cone_min_max.items():
                if at_most and max <= cls:
                    result_lower[cls].add(x)
                elif not at_most and min >= cls:
                    result_lower[cls].add(x)

            # upper
            for x, values in cones.items():
                if (at_most and self[self.decision_attr][x] <= cls) or (
                    not at_most and self[self.decision_attr][x] >= cls
                ):
                    result_upper[cls] = result_upper[cls].union(values)

        return result_lower, result_upper

    def boundary(
        self,
        at_most: bool = False,
        criteria: list[Criterion] = [],
    ) -> pd.Series:
        """Calculates boundaries (doubtful regions)

        :param at_most: If ``True``, the method would return 'at most' boundaries,
        'at least' otherwise, defaults to ``False``

        :return: a `pandas.Series` object, indexed with alternatives names,
        valued with sets of alternatives
        """
        lower_approximation, upper_approximation = self.class_approximation(
            at_most=at_most, criteria=criteria
        )
        return upper_approximation - lower_approximation

    def classification_quality(self, criteria: list[Criterion] = []) -> numeric:
        """Calculates classification quality.

        :return: a numeric value from the [0, 1] interval
        """
        boundary = self.boundary(at_most=False, criteria=criteria).tolist()
        quality = (len(self) - len(set.union(*boundary))) / len(self)

        assert 0 <= quality <= 1
        return quality

    def reductors(self) -> list[set[Criterion]]:
        """Returns list of all reductors (minimal sets of criteria).

        :return: a `list` object with sets of `Criterion` objects.
        """
        attributes = {attr for attr in self.columns.values if not attr.is_decision_attr}
        classification_quality = self.classification_quality()
        reductors: list[set[Criterion]] = []

        reductors_queue = [attributes]
        while reductors_queue:
            attr_set = reductors_queue.pop()
            for attr in attr_set:
                new_attr_set = attr_set - {attr}
                if new_attr_set in reductors:
                    continue

                if math.isclose(
                    classification_quality,
                    self.classification_quality(list(new_attr_set)),
                ):
                    reductors_queue.append(new_attr_set)

                elif attr_set not in reductors:
                    reductors = [reduct for reduct in reductors if not attr_set.issubset(reduct)]
                    reductors.append(attr_set)

        return reductors

    def core(self) -> set[Criterion]:
        """Returns a core, which is an intersection of all reductors.

        :return: a `set` object with `Criterion` objects inside
        """
        return set.intersection(*self.reductors())
