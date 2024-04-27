def calculate_rule_score(covered_objects: set, objects_from_class: set) -> float:
    """Calculate the score of the rule for one decision class value."""
    return len(covered_objects & objects_from_class) ** 2 / (len(objects_from_class) * len(covered_objects))


def calculate_rule_credibility(covered_objects: set, objects_from_class: set) -> float:
    """Calculate the credibility of the rule for one decision class value."""
    return len(covered_objects & objects_from_class) / len(covered_objects)


def calculate_rule_relative_strength(covered_objects: set, objects_from_class: set) -> float:
    """Calculate the relative strength of the rule for one decision class value."""
    return len(covered_objects & objects_from_class) / len(objects_from_class)


def calculate_support(covered_objects_by_condition: set, covered_objects_by_decision: set) -> float:
    """Calculate the support of the rule."""
    return len(covered_objects_by_condition & covered_objects_by_decision)


def calculate_strength(covered_objects_by_condition: set, covered_objects_by_decision: set, all_objects: set) -> float:
    """Calculate the strength of the rule."""
    return calculate_support(covered_objects_by_condition, covered_objects_by_decision) / len(all_objects)


def calculate_certainty_factor(covered_objects_by_condition: set, covered_objects_by_decision: set) -> float:
    """Calculate the certainty factor of the rule."""
    return calculate_support(covered_objects_by_condition, covered_objects_by_decision) / len(
        covered_objects_by_condition
    )


def calculate_coverage_factor(covered_objects_by_condition: set, covered_objects_by_decision: set) -> float:
    """Calculate the coverage factor of the rule."""
    return calculate_support(covered_objects_by_condition, covered_objects_by_decision) / len(
        covered_objects_by_decision
    )
