"""Tests for the MeSH domain filter (ADR-0001 domains)."""

from meridian.corpus.mesh import DOMAINS, classify_domains, in_scope


def test_domains_are_the_three_adr0001_domains() -> None:
    assert set(DOMAINS) == {"cardiology", "endocrinology", "oncology"}


def test_specific_descriptor_matches_via_substring() -> None:
    # Long-tail descriptor names still match their domain keyword.
    assert classify_domains(["Carcinoma, Non-Small-Cell Lung"]) == {"oncology"}
    assert classify_domains(["Myocardial Infarction"]) == {"cardiology"}
    assert classify_domains(["Diabetes Mellitus, Type 2"]) == {"endocrinology"}


def test_matching_is_case_insensitive() -> None:
    assert classify_domains(["HEART FAILURE"]) == {"cardiology"}


def test_multi_domain_record() -> None:
    domains = classify_domains(["Diabetic Cardiomyopathies", "Insulin Resistance"])
    assert domains == {"cardiology", "endocrinology"}


def test_out_of_scope_returns_empty() -> None:
    assert classify_domains(["Malaria", "Photosynthesis"]) == frozenset()
    assert in_scope(["Malaria"]) is False


def test_in_scope_true_for_domain_hit() -> None:
    assert in_scope(["Breast Neoplasms"]) is True
