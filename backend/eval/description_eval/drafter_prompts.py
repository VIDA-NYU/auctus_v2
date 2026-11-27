"""Drafter prompt variants for the record-level draft loop (drafter_loop.py).

Variants are never edited once they've been scored; a tweak gets a new name (V1 -> V1b), so every
round stays reproducible from the LLM cache. Every variant gets the same inputs and the same 120-200
word limit (a longer draft would win on recall by saying more).

Two factors:
  structure     open (V0) | ordered checklist (V1) | fact sheet, then write (V2) | story first (V3)
  prompt style  plain (V0) | StraTyper-style named rules with MUST / do-not lines (V1-V3)
"""

from __future__ import annotations

# Shared tail: the inputs and the output format.
INPUTS = """Record title: {title}

{documentation}Tables in the record (profiled over every row):
{tables}
"""

OUTPUT = """Write 120 to 200 words of plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""

# StraTyper-style rules (named, bold, MUST / do-not), shared by V1-V3.
RULES = """Your draft must follow these key rules:

* **Use all available information**: the README / codebook for what the data means, who made it, why
  and how; the table profiles for variables, ranges, dates and size; the title for the study's name.
  Expand an abbreviation only when the inputs say what it stands for.

* **Be specific**: name the actual variables, places, years and counts. "2001 to 2021" beats "two
  decades"; "hershey, lindt and neither choice indicators" beats "preference data".

* **Handle missing information**: if the inputs don't say who made it, why or how, leave it out. Do NOT
  guess, and do NOT write that something is "not specified".

* **No filler**: no "comprehensive", "valuable for researchers", or lists of possible uses.
"""

V0 = """Draft the description for a dataset record in NYU's research data repository, for a curator to edit.

Record title: {title}

{documentation}Tables in the record (profiled over every row):
{tables}

Write 120 to 200 words: what the data is and covers (variables, place, time span, size), who
produced it and why, and how it was collected, as far as the inputs say. State only what the
inputs support; leave out anything they don't say rather than guessing. Plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""

V1 = """You are an expert research data curator at NYU's data repository. Draft the description of a
dataset record for a curator to edit.

""" + RULES + """
* **Cover these, in this order**: (1) what the data is and what one row or item represents; (2) the
  main variables, with units, categories or ranges; (3) where and when it covers; (4) how big it is
  (tables, rows, files); (5) who produced it and why (the study or project and its aim); (6) how it was
  collected; (7) known gaps or caveats.

""" + INPUTS + "\n" + OUTPUT

V2_SHEET = """You are an expert research data curator at NYU's data repository. Fill in a fact sheet for a
dataset record from the inputs below.

Evaluation steps:

1. **Analyze all evidence**: read the title, the README / codebook (if any) and every table profile.
2. **Fill each field** with what the inputs state, and name its source: "readme", "data" or "title".
3. **Be specific**: actual variable names with units, categories or ranges; actual places, years, counts.
4. **Handle no match**: if the inputs don't say, the field MUST be null. Do NOT guess or fill in what
   is typical for this kind of study.

""" + INPUTS + """
Return JSON only:
{{"topic": {{"value": "...", "source": "..."}},
  "unit_of_observation": {{"value": "what one row or item is", "source": "..."}},
  "variables": {{"value": "the main variables, with units / categories / ranges", "source": "..."}},
  "place": {{"value": "...", "source": "..."}},
  "time_span": {{"value": "...", "source": "..."}},
  "size": {{"value": "tables, rows, files", "source": "..."}},
  "tables": {{"value": "what each table holds and how they relate", "source": "..."}},
  "creators": {{"value": "who produced it", "source": "..."}},
  "purpose": {{"value": "why: the study or project and its aim", "source": "..."}},
  "collection_method": {{"value": "how it was collected", "source": "..."}},
  "funding": {{"value": "...", "source": "..."}},
  "related_publications": {{"value": "...", "source": "..."}},
  "caveats": {{"value": "known gaps, limits or quality notes", "source": "..."}}}}
Use null in place of any object the inputs don't support.
"""

V2_WRITE = """You are an expert research data curator at NYU's data repository. Write the description of a
dataset record for a curator to edit, from this fact sheet ONLY.

Record title: {title}

Fact sheet:
{facts}

""" + RULES + """
* **Use only the fact sheet**: cover its non-null fields in this order: what the data is and what one
  row represents, variables, place and time, size and tables, who made it and why, how it was
  collected, caveats. Do NOT add anything the sheet doesn't say.

""" + OUTPUT

V3 = """You are an expert research data curator at NYU's data repository. Draft the description of a
dataset record for a curator to edit. A record often holds several tables; the description must tell
one story about why they belong together, not describe each table in turn.

""" + RULES + """
* **Tell the story first**: open with the study or project behind the record and its aim (who did it
  and why, if the inputs say). Then say what each table contributes, one clause each. Then the coverage:
  main variables, place, time span, size. Then how the data was collected, and any caveats.

""" + INPUTS + "\n" + OUTPUT

# ---- Round 2 (after round 1 on dev, 2026-11-26) ----
# Round 1: V0 won recall (44.9%); every structured prompt gained ~5 pts on data claims and lost ~7 on
# context claims (findings, scope, other materials), which are 41% of what curators write. V2's fact
# sheet won Koesten (+12). V1 overshot the word limit (206). So: keep V0's open narrative and add the
# context curators write (V4); give the fact sheet context fields (V5); or split narrative and data
# into two paragraphs (V6). All get a hard cap.

CONTEXT = """Curators also write about the study around the data, so include it when the inputs say: what
the study found or concluded, what else the record holds besides the tables (questionnaires, code,
reports, images), and the scope of the project."""

CAP = """Write 120 to 200 words, never more than 200, plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""

V4 = """Draft the description for a dataset record in NYU's research data repository, for a curator to edit.

Record title: {title}

{documentation}Tables in the record (profiled over every row):
{tables}

Cover what the data is and covers (variables, place, time span, size), who produced it and why, and
how it was collected, as far as the inputs say. """ + CONTEXT + """ State only what the inputs support;
leave out anything they don't say rather than guessing, and never write that something is "not
specified".

""" + CAP

V5_SHEET = V2_SHEET.replace(
    '''  "caveats": {{"value": "known gaps, limits or quality notes", "source": "..."}}}}''',
    '''  "caveats": {{"value": "known gaps, limits or quality notes", "source": "..."}},
  "findings": {{"value": "what the study found or concluded", "source": "..."}},
  "other_materials": {{"value": "what the record holds besides the tables (questionnaires, code, reports)", "source": "..."}},
  "scope": {{"value": "the project's scope or context", "source": "..."}}}}''')

V5_WRITE = """You are an expert research data curator at NYU's data repository. Write the description of a
dataset record for a curator to edit, from this fact sheet ONLY.

Record title: {title}

Fact sheet:
{facts}

""" + RULES + """
* **Use only the fact sheet**: open with the study, its aim and who made it; then findings, scope and
  other materials; then what the data is and what one row represents, variables, place and time, size;
  then how it was collected and caveats. Skip null fields. Do NOT add anything the sheet doesn't say.

""" + CAP

V6 = """You are an expert research data curator at NYU's data repository. Draft the description of a
dataset record for a curator to edit, in two short paragraphs.

""" + RULES + """
* **Paragraph 1, the study**: the study or project behind the record, who did it and why, how the data
  was collected, what it found, and what else the record holds besides the tables.
* **Paragraph 2, the data**: what one row or item is, the main variables (with units, categories or
  ranges), where and when it covers, and how big it is.
Leave out any part the inputs don't support.

""" + INPUTS + "\n" + CAP

# ---- Round 3 (after round 2 on dev) ----
# Round 2: only 58.6% of curator claims are in the drafter's inputs at all; 85% of V0's misses aren't,
# so a prompt can add a few points at most. Of the misses that ARE in the inputs, a cluster is access
# and documentation (who to email for restricted data, "a data dictionary is provided", "transcripts
# are anonymized"), which no prompt asks for. V4's "context" line invited speculation about uses
# ("for linguistic analysis, historical research...") and raised flagged claims; V5's two-stage
# writer broke the 200-word cap on 7 of 20 records (max 280).

ACCESS = """Also say how the data can be accessed (open, restricted, who to contact) and what documentation
comes with it (codebook, data dictionary, README, anonymization), when the inputs say."""

V7 = """Draft the description for a dataset record in NYU's research data repository, for a curator to edit.

Record title: {title}

{documentation}Tables in the record (profiled over every row):
{tables}

Write 120 to 200 words: what the data is and covers (variables, place, time span, size), who
produced it and why, and how it was collected, as far as the inputs say. """ + ACCESS + """ State only
what the inputs support; leave out anything they don't say rather than guessing. Never write that
something is "not specified", and don't list possible uses of the data.

Never more than 200 words, plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""

V8_SHEET = V2_SHEET.replace(
    '''  "caveats": {{"value": "known gaps, limits or quality notes", "source": "..."}}}}''',
    '''  "caveats": {{"value": "known gaps, limits or quality notes", "source": "..."}},
  "access": {{"value": "open or restricted, who to contact", "source": "..."}},
  "documentation": {{"value": "codebook, data dictionary, README, anonymization", "source": "..."}}}}''')

V8_WRITE = """You are an expert research data curator at NYU's data repository. Write the description of a
dataset record for a curator to edit, from this fact sheet ONLY.

Record title: {title}

Fact sheet:
{facts}

""" + RULES + """
* **Use only the fact sheet**, skipping null fields, in this order: what the data is and what one row
  represents; who made it and why; how it was collected; variables; place and time; size and tables;
  access and documentation; caveats. Do NOT add anything the sheet doesn't say.

* **Hard limit, 200 words**: if the sheet holds more than fits, keep what the data is, who made it and
  why, the main variables, place and time, and drop the rest. Never go over 200 words.

Plain prose, no headings.

Return JSON only: {{"description": "..."}}
"""

# ---- Round 4: inputs, not wording (2026-11-26) ----
# V9: V0's prompt, drafted from AutoDDG's own per-file descriptions instead of raw table profiles (can a
#     record mode sit on top of what AutoDDG already writes?).
# V10: V0 + the provenance sheet from the record's linked paper(s) (papers.py: full text, data-relevant
#     sections only). Judged with the paper excerpts in the evidence (--with-paper).

# V0b: V0 + one sentence, for the no-README arm, where V0 writes "the methodology is not specified in
# the available metadata" lines that the judge flags (half of that arm's contradicted claims, 11-26).
V0b = V0.replace("leave out anything they don't say rather than guessing.",
                 "leave out anything they don't say rather than guessing, and never write that something is \"not specified\" or \"not available\".")

V9 = V0.replace("Tables in the record (profiled over every row):",
                "Descriptions of the record's tables (written by AutoDDG, one per table):")

V10 = V0.replace("{documentation}Tables in the record", "{documentation}{paper}Tables in the record")

# ---- Round 5 (after round 4) ----
# V10 (+ paper sheet) on 12 records: recall +7.2 (CI +0.8 to +14.7), questions +9.1, flagged +4.4.
# The new failure mode: the draft describes the PAPER's data as the deposit's ("the dataset consists of
# restricted-use administrative records from the Alaska Trauma Registry" when the record holds derived
# population tables and the raw data must be requested). V10b says which is which.
V10b = V10.replace("{documentation}{paper}Tables in the record",
                   "{documentation}{paper}The paper describes the whole study; this record may hold only part of its "
                   "data. Describe the tables below as the record's data, and use the paper only for who made the "
                   "data, why, and how it was collected. If the paper names data that is not in these tables, do not "
                   "say the record contains it.\n\nTables in the record")

# ---- Round 6 ----
# V10b fixed the conflation (flagged 7.5% -> 2.8%) but "use the paper ONLY for who, why, how" also cut
# the findings / scope claims V10 got right (context recall 35% -> 21%). V10c keeps the guard and lets
# the paper supply what the study found and its scope.
V10c = V10.replace("{documentation}{paper}Tables in the record",
                   "{documentation}{paper}The paper describes the whole study; this record may hold only part of its "
                   "data. Describe the tables below as the record's data. Use the paper for who made the data, why, how "
                   "it was collected, the study's scope, and what it found. Never say the record contains data that is "
                   "not in these tables.\n\nTables in the record")

# ---- Length control (2026-11-27) ----
# V2 (fact sheet) beat V0 on Koesten by 12 points but also wrote longer drafts (187 vs 141 words), and a
# longer draft answers more questions. V0L is V0 asked for 180-200 words, to see how much is length.
V0L = V0.replace("Write 120 to 200 words:", "Write 180 to 200 words:")

VARIANTS: dict[str, dict] = {
    "V0": {"stages": [V0], "structure": "open", "style": "plain"},
    "V1": {"stages": [V1], "structure": "ordered", "style": "rules"},
    "V2": {"stages": [V2_SHEET, V2_WRITE], "structure": "fact sheet", "style": "rules"},
    "V3": {"stages": [V3], "structure": "story first", "style": "rules"},
    "V4": {"stages": [V4], "structure": "open + context", "style": "plain"},
    "V5": {"stages": [V5_SHEET, V5_WRITE], "structure": "fact sheet + context", "style": "rules"},
    "V6": {"stages": [V6], "structure": "two paragraphs", "style": "rules"},
    "V7": {"stages": [V7], "structure": "open + access", "style": "plain"},
    "V8": {"stages": [V8_SHEET, V8_WRITE], "structure": "fact sheet + access, capped", "style": "rules"},
    "V0b": {"stages": [V0b], "structure": "open", "style": "plain"},
    "V0L": {"stages": [V0L], "structure": "open, 180-200 words", "style": "plain"},
    "V9": {"stages": [V9], "structure": "open", "style": "plain", "inputs": "autoddg"},
    "V10": {"stages": [V10], "structure": "open", "style": "plain", "inputs": "paper"},
    "V10b": {"stages": [V10b], "structure": "open", "style": "plain", "inputs": "paper"},
    "V10c": {"stages": [V10c], "structure": "open", "style": "plain", "inputs": "paper"},
}
