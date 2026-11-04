"""Prompt templates and JSON schemas. General rules only: nothing here is derived from the answer key."""

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "match_id": {"type": "string", "description": "a candidate id, or NEW"},
        "confidence": {"type": "number", "description": "probability (0-1) that this answer is right"},
        "current_name": {"type": "string", "description": "the entity's current official/brand name"},
        "reason": {"type": "string"},
    },
    "required": ["match_id", "confidence", "current_name", "reason"],
    "additionalProperties": False,
}

MATCH_ORG_SYSTEM = """You resolve organization records (startups, companies, investors) for a venture-data database.
Decide whether the INPUT organization is the same real-world organization as exactly one CANDIDATE from the database, or NEW (not in the database).

Same organization:
- the same company/firm whose name differs only by legal suffix (Inc, Ltd, LLC, GmbH), capitalization, punctuation or abbreviation;
- the same organization after a rename, rebrand or spin-out that kept the business: the database stores the CURRENT name, the input may use a former one (e.g. an old corporate name, an old regional brand of a fund that now operates under a new name).

Different organizations (answer NEW unless another candidate fits):
- unrelated organizations that share or resemble a name (very common: short or generic names such as "Apex", "Nova" or "Summit" belong to many different companies);
- parent vs subsidiary, sister companies under one group, a holding vs an operating company, an acquirer vs a company it bought that kept operating as its own brand.

- two organizations doing similar things (same niche, same sector, same kind of fund) under different names: they are competitors, not one organization.

How to decide: compare what the organization does, its sector, customers, geography, founders, founding year and size in the input description and web results against each candidate's description and links.
- A matching name alone is never enough: check that the businesses agree. A vague candidate description that merely does not contradict the input is weak evidence when the name is generic.
- Match across DIFFERENT names only if you actually know the input name is a former/alternate name of that candidate, or a candidate's profile slug or a web result shows it. Similar descriptions alone never justify it.
- A candidate's own links are just its record. They count as evidence only when a web result about the INPUT points to the same website/profile; then it is strong evidence, after you check the result really is about the input organization.
- If the input's own website (a fetched homepage whose title fits the input) is a DIFFERENT site from the candidate's website, they are most likely different organizations that share a name, unless one site redirects to the other or it is clearly a regional site of the same brand. "Websites fetched" may include unverified guesses: trust a site only if its title/description fits the input.
Set confidence to your honest probability of being right. current_name: the input entity's current official name (its own brand, not a parent's)."""

MATCH_PERSON_SYSTEM = """You resolve person records for a venture-data database.
The INPUT is a person named in a funding-news article excerpt. Each CANDIDATE is a database person with the same or a similar name; the candidate description is their CURRENT profile headline (title at company, location), which can be years newer than the article.
Decide whether the INPUT is the same human as exactly one candidate, or NEW (not in the database).

- Names are shared by many people. A same-name candidate is a match only if something ties them to the article's person: the same organization or one clearly linked to it (acquirer, renamed company, spin-out, portfolio/fund relationship), a career history in web results that contains the article's organization, the same LinkedIn profile URL appearing in results about the article's person, or a consistent role + field + location story.
- People change jobs: a different current employer does NOT rule out a match if web results show the person moved there (e.g. a founder whose startup was acquired, an investor who changed funds).
- If the candidate works in an unrelated field/location and nothing links them to the article's organization, answer NEW.
Set confidence to your honest probability of being right. current_name: the person's full name as commonly written."""

ORG_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "current_name": {"type": "string"},
        "other_names": {"type": "array", "items": {"type": "string"},
                        "description": "former or alternate names you actually know; may be empty"},
        "official_domains": {"type": "array", "items": {"type": "string"},
                             "description": "the entity's own website domain(s) if you know them; may be empty"},
        "search_query": {"type": "string"},
    },
    "required": ["current_name", "other_names", "official_domains", "search_query"],
    "additionalProperties": False,
}

ORG_PROFILE_SYSTEM = """You help look up ONE organization (startup, company or investor) described by a name and a short description.
Return:
- current_name: its current official/brand name (it may have been renamed; keep the input name if unsure);
- other_names: former or alternate names you actually know for THIS organization (not parents, not namesakes); empty if none;
- official_domains: its own website domain(s), e.g. "example.com", only if you are fairly sure they belong to THIS organization and not a namesake; at most 3; empty if unsure. They will be verified by fetching the site;
- search_query: a short Google query (3-7 words) that finds this organization's official website and not namesakes: its name plus one or two distinguishing words (sector, city or country)."""

PERSON_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "organization": {"type": "string", "description": "main organization the person is tied to in the article"},
        "role": {"type": "string"},
        "other_organizations": {"type": "array", "items": {"type": "string"}},
        "location": {"type": "string"},
        "summary": {"type": "string", "description": "one sentence: who this person is according to the article"},
    },
    "required": ["organization", "role", "other_organizations", "location", "summary"],
    "additionalProperties": False,
}

PERSON_CONTEXT_SYSTEM = """Extract facts about ONE named person from a news excerpt. Use only the excerpt. If a field is unknown, return an empty string. organization is the company or fund the person works for or founded (as named in the excerpt), not the company they invested in, unless they work there."""

SAME_SCHEMA = {
    "type": "object",
    "properties": {
        "same": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["same", "confidence", "reason"],
    "additionalProperties": False,
}

SAME_SYSTEM = """Two INPUT records were collected from different sources. Decide whether they describe the same real-world {kind}.
Names can be written differently (case, legal suffixes, abbreviations, nicknames) and descriptions are written independently, so judge by substance: what it does / who they are, organization, geography, people, dates.
Records that only share a name but describe different businesses or different people are NOT the same. Parent and subsidiary, or sister companies, are NOT the same.
confidence = your honest probability that your answer is right."""

LINK_FIELD = {
    "type": "object",
    "properties": {"choice": {"type": "integer", "description": "option number, or -1 for none"},
                   "confidence": {"type": "number"}},
    "required": ["choice", "confidence"],
    "additionalProperties": False,
}


def link_schema(fields: list[str]) -> dict:
    return {"type": "object", "properties": {f: LINK_FIELD for f in fields}, "required": fields,
            "additionalProperties": False}


LINK_SYSTEM = """You pick the official web presence of ONE {kind} from candidate URLs that were found in Google results or on the entity's own website.
For each field choose the option number of the entity's OWN current canonical page, or -1 if no option is clearly it.

Rules:
- It must be this exact entity: never a parent company's, acquirer's, investor's, portfolio company's, founder's/employee's, namesake's or a fan/news page. For a person, never their company's page.
- For a renamed entity use the current brand's page.
- domain = the entity's own website (not a social or data site).
- Prefer the page whose title/snippet clearly names this entity and fits its description (sector, location, people). Links found on the entity's own website are strong evidence for linkedin/x/crunchbase if they are the site's own social profiles (not links to partners or portfolio companies).
- Search results are noisy; when two options are equally plausible or nothing fits, answer -1. A wrong answer costs as much as a right one earns, so only pick when you are more likely right than wrong.
confidence = your honest probability that the chosen option is correct (use 0 for -1)."""
