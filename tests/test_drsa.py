import numpy as np
import pytest


def test_class_unions(alternatives, expected_upward_union, expected_downward_union):
    assert alternatives.upward_class_unions == expected_upward_union
    assert alternatives.downward_class_unions == expected_downward_union


def test_cones(alternatives, partial_expected_negative_cones, partial_expected_positive_cones):
    alternatives.dominance_cones()

    assert len(alternatives.negative_cones) == len(alternatives.data)
    assert len(alternatives.positive_cones) == len(alternatives.data)

    for alternative_name, cone in partial_expected_negative_cones.items():
        assert np.array_equal(alternatives.negative_cones[alternative_name].objects, cone)

    for alternative_name, cone in partial_expected_positive_cones.items():
        assert np.array_equal(alternatives.positive_cones[alternative_name].objects, cone)


def test_class_approximations(alternatives, expected_class_unions_at_least, expected_class_unions_at_most):
    alternatives.class_unions()

    assert alternatives.at_least_approximations.keys() == expected_class_unions_at_least.keys()
    assert alternatives.at_most_approximations.keys() == expected_class_unions_at_most.keys()

    for class_value, expected in expected_class_unions_at_least.items():
        print(alternatives.at_least_approximations)
        assert np.array_equal(
            alternatives.at_least_approximations[class_value].lower_approximation,
            expected.lower_approximation,
        )
        assert np.array_equal(
            alternatives.at_least_approximations[class_value].upper_approximation,
            expected.upper_approximation,
        )
        assert np.array_equal(
            alternatives.at_least_approximations[class_value].objects_that_belong,
            expected.objects_that_belong,
        )

    for class_value, expected in expected_class_unions_at_most.items():
        assert np.array_equal(
            alternatives.at_most_approximations[class_value].lower_approximation,
            expected.lower_approximation,
        )
        assert np.array_equal(
            alternatives.at_most_approximations[class_value].upper_approximation,
            expected.upper_approximation,
        )
        assert np.array_equal(
            alternatives.at_most_approximations[class_value].objects_that_belong,
            expected.objects_that_belong,
        )


@pytest.mark.skip(reason="Looking for a data to test this function")
def test_rules_certain_domlem(alternatives, expected_certain_rules):
    alternatives.rules()


@pytest.mark.skip(reason="Looking for a data to test this function")
def test_rules_possible_domlem(alternatives, expected_possible_rules):
    alternatives.rules()
