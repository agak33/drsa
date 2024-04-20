import numpy as np
import pytest

from drsa.alternative_set import Metric


def test_class_unions(alternatives, expected_upward_union, expected_downward_union):
    assert alternatives.upward_class_unions == expected_upward_union
    assert alternatives.downward_class_unions == expected_downward_union


def test_cones(alternatives, partial_expected_negative_cones, partial_expected_positive_cones):
    assert len(alternatives.negative_cones) == len(alternatives.data)
    assert len(alternatives.positive_cones) == len(alternatives.data)

    for alternative_name, cone in partial_expected_negative_cones.items():
        assert np.array_equal(alternatives.negative_cones[alternative_name].objects, cone)

    for alternative_name, cone in partial_expected_positive_cones.items():
        assert np.array_equal(alternatives.positive_cones[alternative_name].objects, cone)


def test_class_approximations(
    alternatives, expected_class_approximations_at_least, expected_class_approximations_at_most
):
    alternatives.class_approximations()

    assert alternatives.at_least_approximations.keys() == expected_class_approximations_at_least.keys()
    assert alternatives.at_most_approximations.keys() == expected_class_approximations_at_most.keys()

    for class_value, expected in expected_class_approximations_at_least.items():
        assert set(alternatives.at_least_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].positive_region) == set(
            expected["positive_region"]
        )
        assert set(alternatives.at_least_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_least_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_least_approximations[class_value].accuracy == expected["accuracy"]

    for class_value, expected in expected_class_approximations_at_most.items():
        assert set(alternatives.at_most_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].positive_region) == set(expected["positive_region"])
        assert set(alternatives.at_most_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_most_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_most_approximations[class_value].accuracy == expected["accuracy"]


@pytest.mark.parametrize(
    ("threshold", "type"),
    (
        (0.001, Metric.COST_TYPE),
        (1.0, Metric.GAIN_TYPE),
    ),
)
def test_class_approximations_threshold(
    alternatives,
    expected_class_approximations_at_least,
    expected_class_approximations_at_most,
    threshold,
    type,
):
    alternatives.class_approximations(threshold=threshold, metric_type=type)

    assert alternatives.at_least_approximations.keys() == expected_class_approximations_at_least.keys()
    assert alternatives.at_most_approximations.keys() == expected_class_approximations_at_most.keys()

    for class_value, expected in expected_class_approximations_at_least.items():
        assert set(alternatives.at_least_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].positive_region) == set(
            expected["positive_region"]
        )
        assert set(alternatives.at_least_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_least_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_least_approximations[class_value].accuracy == expected["accuracy"]

    for class_value, expected in expected_class_approximations_at_most.items():
        assert set(alternatives.at_most_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].positive_region) == set(expected["positive_region"])
        assert set(alternatives.at_most_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_most_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_most_approximations[class_value].accuracy == expected["accuracy"]


def test_class_approximations_cost(
    alternatives,
    expected_class_approximations_at_least_epsilon,
    expected_class_approximations_at_most_epsilon,
):
    alternatives.class_approximations(threshold=0.1, metric_type=Metric.COST_TYPE)

    assert alternatives.at_least_approximations.keys() == expected_class_approximations_at_least_epsilon.keys()
    assert alternatives.at_most_approximations.keys() == expected_class_approximations_at_most_epsilon.keys()

    for class_value, expected in expected_class_approximations_at_least_epsilon.items():
        assert set(alternatives.at_least_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].positive_region) == set(
            expected["positive_region"]
        )
        assert set(alternatives.at_least_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_least_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_least_approximations[class_value].accuracy == expected["accuracy"]

    for class_value, expected in expected_class_approximations_at_most_epsilon.items():
        assert set(alternatives.at_most_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].positive_region) == set(expected["positive_region"])
        assert set(alternatives.at_most_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_most_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_most_approximations[class_value].accuracy == expected["accuracy"]


def test_class_approximations_gain(
    alternatives,
    expected_class_approximations_at_least_mu,
    expected_class_approximations_at_most_mu,
):
    alternatives.class_approximations(threshold=0.9, metric_type=Metric.GAIN_TYPE)

    assert alternatives.at_least_approximations.keys() == expected_class_approximations_at_least_mu.keys()
    assert alternatives.at_most_approximations.keys() == expected_class_approximations_at_most_mu.keys()

    for class_value, expected in expected_class_approximations_at_least_mu.items():
        assert set(alternatives.at_least_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_least_approximations[class_value].positive_region) == set(
            expected["positive_region"]
        )
        assert set(alternatives.at_least_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_least_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_least_approximations[class_value].accuracy == expected["accuracy"]

    for class_value, expected in expected_class_approximations_at_most_mu.items():
        assert set(alternatives.at_most_approximations[class_value].lower_approximation) == set(
            expected["lower_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].upper_approximation) == set(
            expected["upper_approximation"]
        )
        assert set(alternatives.at_most_approximations[class_value].positive_region) == set(expected["positive_region"])
        assert set(alternatives.at_most_approximations[class_value].boundary) == set(expected["boundary"])
        assert alternatives.at_most_approximations[class_value].quality == expected["quality"]
        assert alternatives.at_most_approximations[class_value].accuracy == expected["accuracy"]


@pytest.mark.skip(reason="Looking for a data to test this function")
def test_rules_certain_domlem(alternatives, expected_certain_rules):
    alternatives.rules()


@pytest.mark.skip(reason="Looking for a data to test this function")
def test_rules_possible_domlem(alternatives, expected_possible_rules):
    alternatives.rules()
