"""Grade a submission against an answer key and say exactly what went wrong.

    python3 grade.py SUBMISSION_DIR                    # visible set, key in this script's folder
    python3 grade.py SUBMISSION_DIR --key DIR --holdout  # holdout set (answer_key_holdout_*, holdout_input_*)

SUBMISSION_DIR holds submission_{organizations,people}.csv (columns as in submission_template_*.csv).
Writes grade_report.md and grade_errors.csv to --out (default SUBMISSION_DIR) and prints a summary.

Scoring, per input row:
  +1  correct: the right existing id, or NEW with exactly the right dup group
  -1  anything else:
        duplicate    NEW given for an existing entity, or a NEW dup group missing a twin
        wrong_merge  the wrong existing id, an existing id given for a new entity, or a NEW dup group too big
        invalid      row missing or match_id blank, match_id not in the DB and not NEW, or dup_of naming an unknown input_id
Links count only on rows that are truly new and called NEW. With k = links in the key (min 1): each correct link
+1/k, each wrong link or a value where the key is blank -1/k, a blank 0; the row's link score is clamped to -1..+1.
Score = total / (rows + truly-new rows), shown as a %; it can go negative.

Standard library only (Python 3.9+).
"""
import argparse
import csv
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from urllib.parse import unquote

TYPES = ["organizations", "people"]
LINKS = {"organizations": ["domain", "linkedin", "crunchbase", "x"],
         "people": ["linkedin", "crunchbase", "x"]}
ERR_COLS = ["set", "type", "input_id", "name", "kind", "cause", "field", "submitted", "expected",
            "submitted_name", "expected_name", "points", "detail", "bucket", "tags"]
CAUSES = {  # cause -> (kind, what it means), in report order
    "wrong_existing": ("wrong_merge", "matched to the wrong existing record"),
    "new_matched_existing": ("wrong_merge", "a new entity matched to an existing record"),
    "group_too_big": ("wrong_merge", "NEW rows grouped with rows that are a different entity"),
    "exists_called_new": ("duplicate", "called NEW but the entity is already in the DB"),
    "group_too_small": ("duplicate", "NEW twin rows not linked with dup_of"),
    "missing_row": ("invalid", "no row, or a blank match_id, for this input_id"),
    "bad_match_id": ("invalid", "match_id is not an id in the DB and not NEW"),
    "bad_dup_of": ("invalid", "dup_of is not an input_id in this file"),
    "link_wrong": ("link", "link differs from the key"),
    "link_extra": ("link", "link given where the entity has none"),
    "link_missed": ("link", "link left blank (0 points, recall only)"),
    "unknown_row": ("note", "input_id not in the input file (ignored)"),
    "repeated_row": ("note", "input_id appears more than once (first row used)"),
}


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(fh)]


def norm(field, v):
    v = unicodedata.normalize("NFC", unquote(v.strip()))
    if not v:
        return ""
    v = v.lower()
    v = re.sub(r"^[a-z][a-z0-9+.-]*://", "", v)
    v = re.split(r"[?#]", v)[0]
    v = re.sub(r"^www\.", "", v)
    v = re.sub(r"^[a-z]{2}\.linkedin\.com/", "linkedin.com/", v)
    v = re.sub(r"^(mobile\.)?twitter\.com/", "x.com/", v)
    return v.rstrip("/")


def groups(ids, edges):
    """Union-find: id -> frozenset of ids joined through edges."""
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b in edges:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    out = defaultdict(set)
    for i in ids:
        out[find(i)].add(i)
    return {i: frozenset(out[find(i)]) for i in ids}


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "n/a"


def grade_type(plural, sub_path, key_path, input_path, db_path, meta_path, set_name):
    key = read_csv(key_path)
    names = {r["input_id"]: r["name"] for r in read_csv(input_path)}
    db = {r["id"]: r["name"] for r in read_csv(db_path)}
    meta = {r["input_id"]: r for r in read_csv(meta_path)} if meta_path and os.path.exists(meta_path) else {}
    fields = LINKS[plural]
    order = [r["input_id"] for r in key]
    truth = {r["input_id"]: r for r in key}
    errors = []

    def err(iid, cause, points, field="", submitted="", expected="", detail=""):
        def label(v):
            return db.get(v, names.get(v, "")) if v else ""
        m = meta.get(iid, {})
        errors.append({"set": set_name, "type": plural, "input_id": iid, "name": names.get(iid, ""),
                       "kind": CAUSES[cause][0], "cause": cause, "field": field, "submitted": submitted,
                       "expected": expected, "submitted_name": label(submitted) if not field else "",
                       "expected_name": label(expected) if not field else "", "points": round(points, 3),
                       "detail": detail, "bucket": m.get("bucket", ""), "tags": m.get("tags", "")})

    sub = {}
    if os.path.exists(sub_path):
        for r in read_csv(sub_path):
            iid = r.get("input_id", "")
            if iid not in truth:
                err(iid, "unknown_row", 0)
            elif iid in sub:
                err(iid, "repeated_row", 0)
            else:
                sub[iid] = r

    def sub_match(iid):
        m = sub.get(iid, {}).get("match_id", "")
        return "NEW" if m.upper() == "NEW" else m

    true_new = [i for i in order if truth[i]["match_id"] == "NEW"]
    true_group = groups(true_new, [(i, truth[i]["dup_of"]) for i in true_new if truth[i]["dup_of"]])
    sub_new = [i for i in order if i in sub and sub_match(i) == "NEW"]
    sub_group = groups(sub_new, [(i, sub[i].get("dup_of", "")) for i in sub_new if sub[i].get("dup_of")])

    def show(group):
        return "{" + ", ".join(f'{i} "{names.get(i, "")}"' for i in sorted(group)) + "}"

    decision, links, ok = 0, 0.0, set()
    lstat = {f: Counter() for f in fields}
    for iid in order:
        t, m = truth[iid], sub_match(iid)
        dup = sub.get(iid, {}).get("dup_of", "")
        if not m:
            err(iid, "missing_row", -1, expected=t["match_id"])
        elif m != "NEW" and m not in db:
            err(iid, "bad_match_id", -1, submitted=m, expected=t["match_id"])
        elif m == "NEW" and dup and dup not in truth:
            err(iid, "bad_dup_of", -1, submitted=dup, detail=f"dup_of {dup!r} is not an input_id in this file")
        elif t["match_id"] != "NEW":
            if m == t["match_id"]:
                ok.add(iid)
            elif m == "NEW":
                err(iid, "exists_called_new", -1, submitted="NEW", expected=t["match_id"])
            else:
                err(iid, "wrong_existing", -1, submitted=m, expected=t["match_id"])
        elif m != "NEW":
            err(iid, "new_matched_existing", -1, submitted=m, expected="NEW")
        else:
            mine, want = sub_group[iid], true_group[iid]
            if mine == want:
                ok.add(iid)
            elif mine - want:
                err(iid, "group_too_big", -1, detail=f"your group {show(mine)}; correct group {show(want)}")
            else:
                err(iid, "group_too_small", -1, detail=f"your group {show(mine)}; correct group {show(want)}")
        decision += 1 if iid in ok else -1

        if t["match_id"] == "NEW" and iid in sub and m == "NEW":
            k = max(1, sum(bool(t[f]) for f in fields))
            row = 0.0
            for f in fields:
                yours, want = sub[iid].get(f, ""), t[f]
                ny, nw = norm(f, yours), norm(f, want)
                lstat[f]["key"] += bool(nw)
                lstat[f]["given"] += bool(ny)
                if ny and ny == nw:
                    row += 1 / k
                    lstat[f]["correct"] += 1
                elif ny and nw:
                    row -= 1 / k
                    err(iid, "link_wrong", -1 / k, f, yours, want)
                elif ny:
                    row -= 1 / k
                    err(iid, "link_extra", -1 / k, f, yours, "")
                elif nw:
                    err(iid, "link_missed", 0, f, "", want)
            links += max(-1.0, min(1.0, row))

    existing = [i for i in order if truth[i]["match_id"] != "NEW"]
    called_new = [i for i in order if i in sub and sub_match(i) == "NEW"]
    kinds = Counter(e["kind"] for e in errors if e["kind"] not in ("link", "note"))
    res = {"type": plural, "rows": len(order), "new_rows": len(true_new), "max": len(order) + len(true_new),
           "decision": decision, "links": links, "total": decision + links,
           "duplicates": kinds["duplicate"], "wrong_merges": kinds["wrong_merge"], "invalid": kinds["invalid"],
           "match_acc": (sum(i in ok for i in existing), len(existing)),
           "new_prec": (sum(truth[i]["match_id"] == "NEW" for i in called_new), len(called_new)),
           "new_rec": (sum(i in called_new for i in true_new), len(true_new)),
           "links_by_field": lstat, "errors": errors, "meta": meta, "ok": ok, "order": order}
    return res


def summary_lines(r):
    lines = [f"score {pct(r['total'], r['max'])}  ({r['total']:.2f} / {r['max']}: decisions {r['decision']:+d}, "
             f"links {r['links']:+.2f})",
             f"duplicates {r['duplicates']}, wrong merges {r['wrong_merges']}, invalid {r['invalid']}",
             f"match accuracy {pct(*r['match_acc'])} ({r['match_acc'][0]}/{r['match_acc'][1]}), "
             f"NEW precision {pct(*r['new_prec'])} ({r['new_prec'][0]}/{r['new_prec'][1]}), "
             f"NEW recall {pct(*r['new_rec'])} ({r['new_rec'][0]}/{r['new_rec'][1]})",
             "links (precision / recall): " + ", ".join(
                 f"{f} {pct(s['correct'], s['given'])} / {pct(s['correct'], s['key'])}"
                 for f, s in r["links_by_field"].items())]
    return lines


def error_line(e):
    nm = lambda i, n: f'{i} "{n}"' if n else i  # noqa: E731
    head = f'`{e["input_id"]}` "{e["name"]}"'
    c = e["cause"]
    if c == "wrong_existing":
        return f"{head}: matched {nm(e['submitted'], e['submitted_name'])}, correct is {nm(e['expected'], e['expected_name'])}"
    if c == "new_matched_existing":
        return f"{head}: matched {nm(e['submitted'], e['submitted_name'])}, but this entity is not in the DB (NEW)"
    if c == "exists_called_new":
        return f"{head}: called NEW, but it exists as {nm(e['expected'], e['expected_name'])}"
    if c in ("group_too_big", "group_too_small", "bad_dup_of"):
        return f"{head}: {e['detail']}"
    if c == "missing_row":
        return f"{head}: no answer submitted (correct: {nm(e['expected'], e['expected_name'])})"
    if c == "bad_match_id":
        return f"{head}: match_id {e['submitted']!r} is not in the DB (correct: {nm(e['expected'], e['expected_name'])})"
    if c == "link_wrong":
        return f"{head}: {e['field']} yours {e['submitted']}, expected {e['expected']}"
    if c == "link_extra":
        return f"{head}: {e['field']} yours {e['submitted']}, expected blank (the entity has none)"
    if c == "link_missed":
        return f"{head}: {e['field']} blank, expected {e['expected']}"
    return f"{head}: {CAUSES[c][1]}"


def report(results, set_name):
    md = [f"# Grade report ({set_name} set)", "",
          "| type | score | total / max | decisions | links | duplicates | wrong merges | invalid | match acc | "
          "NEW prec | NEW rec |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['type']} | {pct(r['total'], r['max'])} | {r['total']:.2f} / {r['max']} | {r['decision']:+d} | "
                  f"{r['links']:+.2f} | {r['duplicates']} | {r['wrong_merges']} | {r['invalid']} | "
                  f"{pct(*r['match_acc'])} | {pct(*r['new_prec'])} | {pct(*r['new_rec'])} |")
    md += ["", "Each row scores +1 if right and -1 if wrong. Links count only on truly-new rows called NEW. "
               "`grade_errors.csv` has every error below as one row."]
    for r in results:
        errs = r["errors"]
        md += ["", f"## {r['type']}", ""] + [f"- {x}" for x in summary_lines(r)]
        lf = Counter(e["field"] for e in errs if e["cause"] in ("link_wrong", "link_extra"))
        cc = Counter(e["cause"] for e in errs)
        pattern = ", ".join(f"{c} {n}" for c, n in sorted(cc.items(), key=lambda kv: -kv[1]))
        md += ["", f"**Patterns:** {pattern or 'no errors'}."
               + (f" Most link errors: `{lf.most_common(1)[0][0]}` ({lf.most_common(1)[0][1]})." if lf else "")]
        if r["meta"]:
            by = defaultdict(lambda: [0, 0])
            for i in r["order"]:
                b = r["meta"].get(i, {}).get("bucket", "")
                by[b][0] += 1
                by[b][1] += i in r["ok"]
            md += ["", "| bucket | rows | decisions right |", "|---|---|---|"]
            md += [f"| {b} | {n} | {k} ({pct(k, n)}) |" for b, (n, k) in sorted(by.items())]
        for cause, (kind, what) in CAUSES.items():
            hit = [e for e in errs if e["cause"] == cause]
            if hit:
                md += ["", f"### {cause} ({kind}, {len(hit)}): {what}", ""]
                md += [f"- {error_line(e)}" + (f" [bucket {e['bucket']}; tags {e['tags']}]" if e["bucket"] else "")
                       for e in hit]
    return "\n".join(md) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("submission_dir")
    ap.add_argument("--key", default=here, help="folder with the answer key and inputs (default: this script's folder)")
    ap.add_argument("--holdout", action="store_true", help="grade against answer_key_holdout_* / holdout_input_*")
    ap.add_argument("--out", help="where to write grade_report.md and grade_errors.csv (default: submission_dir)")
    a = ap.parse_args()
    set_name = "holdout" if a.holdout else "visible"
    kp, ip = ("answer_key_holdout_", "holdout_input_") if a.holdout else ("answer_key_", "input_")
    db_dir = a.key if os.path.exists(os.path.join(a.key, "existing_organizations.csv")) \
        else os.path.join(a.key, os.pardir, "candidate")
    out = a.out or a.submission_dir
    results = []
    for p in TYPES:
        key_path = os.path.join(a.key, f"{kp}{p}.csv")
        if not os.path.exists(key_path):
            sys.exit(f"missing key file {key_path}")
        sub_path = os.path.join(a.submission_dir, f"submission_{p}.csv")
        if not os.path.exists(sub_path):
            print(f"!! {sub_path} not found: every {p} row counts as missing", file=sys.stderr)
        results.append(grade_type(p, sub_path, key_path, os.path.join(a.key, f"{ip}{p}.csv"),
                                  os.path.join(db_dir, f"existing_{p}.csv"),
                                  os.path.join(a.key, f"answer_meta_{set_name}_{p}.csv"), set_name))

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "grade_report.md"), "w") as fh:
        fh.write(report(results, set_name))
    with open(os.path.join(out, "grade_errors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, ERR_COLS)
        w.writeheader()
        for r in results:
            w.writerows(r["errors"])

    for r in results:
        print(f"\n{r['type']} ({set_name})")
        for x in summary_lines(r):
            print("  " + x)
        cc = Counter(e["cause"] for e in r["errors"] if e["kind"] != "note")
        if cc:
            print("  top errors: " + ", ".join(f"{c} {n}" for c, n in cc.most_common(5)))
    tot, mx = sum(r["total"] for r in results), sum(r["max"] for r in results)
    print(f"\noverall {pct(tot, mx)} ({tot:.2f} / {mx})")
    print(f"details: {os.path.join(out, 'grade_report.md')}, {os.path.join(out, 'grade_errors.csv')}")


if __name__ == "__main__":
    main()
