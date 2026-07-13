"""Domain filtering by MeSH terms (ADR-0001: cardiology, endocrinology, oncology).

PubMed article XML carries MeSH *descriptor names*, not tree numbers, so this
filter classifies a record by matching curated keyword substrings against its
descriptor names. The keyword sets are a deliberate, documented proxy for the
MeSH trees C14 (Cardiovascular Diseases), C19/C18 (Endocrine & Metabolic
Diseases), and C04 (Neoplasms). Exact tree-number filtering via the full MeSH
descriptor database is a future refinement; it would only tighten precision, not
change the pipeline.

Matching is case-insensitive substring containment: a descriptor matches a domain
if any of the domain's keywords appears anywhere in the (lowercased) descriptor
name. This keeps the filter robust to the long tail of specific descriptors
(e.g. "Carcinoma, Non-Small-Cell Lung" matches oncology via "carcinoma").
"""

from __future__ import annotations

from collections.abc import Iterable

# Domain -> curated MeSH descriptor-name keyword substrings (all lowercase).
# Chosen for high precision within each domain's MeSH tree while covering the
# common descriptor long tail.
DOMAIN_MESH_KEYWORDS: dict[str, frozenset[str]] = {
    "cardiology": frozenset(
        {
            "cardio",
            "heart",
            "coronary",
            "myocard",
            "arrhythmi",
            "atrial fibrillation",
            "angina",
            "hypertension",
            "atheroscleros",
            "aortic",
            "endocarditis",
            "pericard",
        }
    ),
    "endocrinology": frozenset(
        {
            "diabet",
            "insulin",
            "thyroid",
            "endocrine",
            "glycemi",
            "obesity",
            "metabolic syndrome",
            "adrenal",
            "pituitary",
            "hyperlipidemi",
            "dyslipidemi",
            "osteoporos",
        }
    ),
    "oncology": frozenset(
        {
            "neoplasm",
            "carcinoma",
            "cancer",
            "tumor",
            "tumour",
            "leukemia",
            "lymphoma",
            "melanoma",
            "sarcoma",
            "malignan",
            "metasta",
            "oncolog",
            "antineoplastic",
        }
    ),
}

DOMAINS: tuple[str, ...] = tuple(DOMAIN_MESH_KEYWORDS)


def classify_domains(mesh_terms: Iterable[str]) -> frozenset[str]:
    """Return the set of ADR-0001 domains a record's MeSH terms fall under.

    A record may match more than one domain (e.g. a study on diabetic
    cardiomyopathy); an empty set means it is out of scope.
    """
    terms_lower = [term.lower() for term in mesh_terms]
    matched: set[str] = set()
    for domain, keywords in DOMAIN_MESH_KEYWORDS.items():
        if any(keyword in term for term in terms_lower for keyword in keywords):
            matched.add(domain)
    return frozenset(matched)


def in_scope(mesh_terms: Iterable[str]) -> bool:
    """Return ``True`` if the record belongs to at least one target domain."""
    return bool(classify_domains(mesh_terms))
