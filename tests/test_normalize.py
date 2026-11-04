import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from er.normalize import (LinkIndex, canon_crunchbase, canon_domain, canon_link, canon_linkedin, canon_x,  # noqa: E402
                          compact_key, compare_key, group_ids, name_key, person_key)


def test_name_key_strips_legal_suffixes_and_case():
    assert name_key("Greycroft LLC") == name_key("greycroft")
    assert name_key("Chesapeake Bay Seed Capital Fund Inc") == "chesapeake bay seed capital fund"
    assert name_key("KOMPAS VC Ltd") == "kompas vc"
    assert name_key("Ørsted") == "orsted"
    assert name_key("Co") == "co"  # never strip a name down to nothing


def test_compact_key_and_person_key():
    assert compact_key("Circle.Co Ltd") == compact_key("circle")  # blocking only; the LLM decides
    assert person_key("Andy Schwartz") == person_key("Andrew Schwartz")
    assert person_key("CHRIS EMERY") == person_key("Christopher Emery")


def test_url_canonicalizers():
    assert canon_domain("https://www.Example.com/about?x=1") == "example.com"
    assert canon_domain("https://www.linkedin.com/company/x") == ""
    assert canon_domain("https://investor.acme.io/news") == "acme.io"
    assert canon_domain("https://en.example.co.uk/") == "example.co.uk"
    assert canon_domain("lifesciences.bigco.com") == "lifesciences.bigco.com"
    assert canon_domain("https://uk.co.uk") == "uk.co.uk"
    assert canon_linkedin("https://www.linkedin.com/company/43207463", "company") == ""
    assert canon_linkedin("https://uk.linkedin.com/company/acme-inc/about/", "company") == \
        "https://www.linkedin.com/company/acme-inc"
    assert canon_linkedin("https://www.linkedin.com/in/jane-doe-123?trk=x", "in") == "https://www.linkedin.com/in/jane-doe-123"
    assert canon_linkedin("https://www.linkedin.com/in/jane", "company") == ""
    assert canon_crunchbase("https://www.crunchbase.com/organization/acme-2/people", "organization") == \
        "https://www.crunchbase.com/organization/acme-2"
    assert canon_crunchbase("https://www.crunchbase.com/funding_round/acme-seed", "organization") == ""
    assert canon_x("https://twitter.com/AcmeHQ") == "https://x.com/AcmeHQ"
    assert canon_x("https://x.com/acmehq/status/123") == ""
    assert canon_x("https://x.com/intent/follow?screen_name=a") == ""
    assert canon_x("https://x.com/home") == ""
    assert canon_link("linkedin", "https://www.linkedin.com/in/a", "people") == "https://www.linkedin.com/in/a"


def test_compare_key_matches_grader_rules():
    assert compare_key("x", "https://twitter.com/Foo/") == compare_key("x", "x.com/foo")
    assert compare_key("linkedin", "https://de.linkedin.com/company/a?x") == compare_key(
        "linkedin", "https://www.linkedin.com/company/a")


def test_link_index_lookup():
    idx = LinkIndex([{"id": "o1", "domain": "acme.com", "linkedin": "https://www.linkedin.com/company/acme",
                      "crunchbase": "", "x": ""}], "organizations")
    assert idx.lookup("domain", "https://www.acme.com/en/") == {"o1"}
    assert idx.lookup("linkedin", "https://fr.linkedin.com/company/acme/") == {"o1"}
    assert idx.lookup("domain", "https://other.com") == set()


def test_group_ids_rep_is_first_in_input_order():
    g = group_ids(["a", "b", "c", "d"], [("c", "a"), ("d", "c")])
    assert g == {"a": ["a", "c", "d"], "b": ["b"]}


def test_db_links_survive_canonicalization():
    """Canonicalizing every DB link must not change it under the grader's comparison."""
    for entity in ("organizations", "people"):
        with open(ROOT / "case-study" / f"existing_{entity}.csv", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            for f in (["domain"] if entity == "organizations" else []) + ["linkedin", "crunchbase", "x"]:
                if r[f]:
                    assert compare_key(f, canon_link(f, r[f], entity)) == compare_key(f, r[f]), (f, r[f])
