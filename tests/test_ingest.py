"""The ingestion checker: does every value in a step appear in the document?

Whether the steps are the right steps is a judgement, and the source sits next
to the output so a person can read one against the other. What a rule can catch
is a number in a step that is nowhere in the document it came from, which is
the shape an invented value has.

The checker is only worth having if its findings are trustworthy, so what is
tested here is mostly the ways it reported a value as invented when the
document contained it.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest_protocol import check  # noqa: E402

CORPUS = ROOT / "data" / "corpus"


def steps(*texts):
    return {"steps": [{"text": t} for t in texts]}


def test_a_value_in_the_source_passes():
    assert check("Incubate at 42°C for 30 secs.", steps("Incubate at 42°C for 30 seconds.")) == []


def test_a_value_not_in_the_source_is_reported():
    """The shape of an invented number. The one real finding on the corpus is
    an annealing temperature the source states as a rule, not a number."""
    problems = check("Anneal 3°C above the primer Tm.", steps("Anneal at 65 °C for 30 seconds."))
    assert any("65" in p for p in problems)


def test_a_thousands_separator_in_the_source_is_not_a_different_number():
    """The source writes "< 1,000 ng" and the step writes "1000 ng". Comparing
    those literally reported the step as inventing a number sitting in the
    document, which is how this checker lost its first argument."""
    assert check("Template DNA | variable | < 1,000 ng", steps("Add no more than 1000 ng.")) == []


def test_a_thousands_separator_in_the_step_is_not_three_numbers():
    """Splitting "250-1,000" on digits gives 250, 1 and 000. Then "000" is
    reported missing from a document that says 1,000, and the lone "1" matches
    almost any document at all, so the check passes for the wrong reason."""
    assert check("Add 250-1,000 μl LB media.", steps("Add 250-1,000 μl of LB media.")) == []


def test_a_range_written_out_in_words_still_matches():
    """The model may write "20 to 30 minutes" where the source wrote "20-30
    mins". Same quantity, so the digits are what is compared."""
    assert check("Thaw on ice (approximately 20-30 mins).", steps("Thaw for 20 to 30 minutes.")) == []


def test_an_empty_step_is_reported():
    assert any("empty" in p for p in check("anything", steps("   ")))


def test_a_document_that_produced_no_steps_is_reported():
    assert check("anything", {"steps": []}) == ["no steps"]


def test_a_timer_label_with_a_digit_is_reported():
    """The label is announced aloud, and a digit reaching Rime is a digit its
    own normaliser expands however it likes."""
    parsed = {"steps": [{"text": "Wait 20 mins.", "timers": [{"label": "20 minute wait", "seconds": 1200}]}]}
    assert any("digit" in p for p in check("Wait 20 mins.", parsed))


def test_a_timer_with_no_length_is_reported():
    parsed = {"steps": [{"text": "Wait.", "timers": [{"label": "a wait", "seconds": 0}]}]}
    assert any("no length" in p for p in check("Wait.", parsed))


@pytest.mark.parametrize("protocol", ["addgene_transformation"])
def test_the_ingested_corpus_still_checks_out_against_its_source(protocol):
    """The one protocol in the corpus this script produced. If a later prompt
    change makes it invent a value, this is where it shows."""
    root = CORPUS / protocol
    parsed = yaml.safe_load((root / "parsed.yaml").read_text())
    assert check((root / "source.md").read_text(), parsed) == []
