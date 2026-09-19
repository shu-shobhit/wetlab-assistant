"""The corpus on disk, checked as data.

A protocol is the one input this product cannot validate at run time: a wrong
number in the YAML is spoken confidently and correctly as a wrong number. So
the loader is strict and these tests are where the files themselves are
checked.

The invariants that used to live here were about spans, hazard counts and
whether every readback span could actually be matched. Those are gone with the
labelling scheme. What replaces them is a check that the derived file and the
source have not drifted apart, which is the new way for the corpus to be quietly
wrong.
"""

from pathlib import Path

import pytest
import yaml

from wetlab import protocol, render

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"

#: Every protocol in the corpus. `handwritten` is authored for the tests and
#: must never appear in an evidence table; the rest are cited.
ALL = ["handwritten", "neb_q5_m0492", "addgene_transformation"]
CITED = ["neb_q5_m0492", "addgene_transformation"]


@pytest.fixture(params=ALL)
def loaded(request):
    return protocol.load(CORPUS, request.param)


def test_it_loads(loaded):
    assert loaded.steps
    assert loaded.citation.strip()


def test_every_step_has_an_id_an_index_and_text(loaded):
    for i, step in enumerate(loaded.steps):
        assert step.id
        assert step.index == i
        assert step.number == i + 1
        assert step.text.strip()


def test_step_ids_are_unique(loaded):
    ids = [s.id for s in loaded.steps]
    assert len(set(ids)) == len(ids)


def test_no_step_carries_the_old_labelling(loaded):
    """spans, hazard classes and irreversible were dropped with the gate.

    Leaving one in a YAML file would be silently ignored by the loader, so the
    file would look like it still meant something it does not.
    """
    for pid in ALL:
        raw = yaml.safe_load((CORPUS / pid / "parsed.yaml").read_text())
        for step in raw["steps"]:
            assert set(step) <= {"id", "text", "timers"}, (pid, step["id"])


def test_every_timer_has_a_label_that_can_be_spoken(loaded):
    """The label is announced aloud when the timer expires, so a digit in it
    reaches Rime's own normaliser, which is the guessing this design removes."""
    for step in loaded.steps:
        for timer in step.timers:
            assert timer.seconds > 0
            assert timer.label.strip()
            assert not any(c.isdigit() for c in timer.label), timer.id


def test_a_cited_protocol_has_its_source_next_to_it():
    for pid in CITED:
        assert (CORPUS / pid / "source.md").exists()
        assert (CORPUS / pid / "CITATION").read_text().strip()


def test_every_step_can_be_spoken_without_leaving_a_digit(loaded):
    """Whether or not the offline pass has been run.

    With a prepared form this checks the committed file; without one it checks
    the detector's fallback. Either way a digit reaching Rime is a digit it
    expands however it likes.
    """
    for step in loaded.steps:
        spoken = loaded.speech_for(step)
        assert spoken.strip(), step.id
        assert not any(c.isdigit() for c in spoken), (step.id, spoken)


def test_every_step_has_terms_to_be_found_by(loaded):
    for step in loaded.steps:
        # A step with no notation in it is found by its own words, so terms may
        # legitimately be empty; a step with a quantity in it may not be.
        if render.detect(step.text):
            assert loaded.terms_for(step), step.id


def test_a_prepared_form_covers_every_step_or_none_of_them(loaded):
    """A partly regenerated spoken.yaml is the failure this catches.

    Half the steps prepared and half falling back to the detector is two
    different voices in one reading, and nothing else would notice.
    """
    if not loaded.spoken:
        pytest.skip("no prepared form generated yet")
    assert set(loaded.spoken) == {s.id for s in loaded.steps}


def test_a_prepared_form_naming_an_unknown_step_is_an_error(tmp_path):
    """Which is what an edited source and a stale derived file look like."""
    root = tmp_path / "p"
    root.mkdir()
    (root / "CITATION").write_text("test")
    (root / "parsed.yaml").write_text(
        "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: 'Do the thing.'\n"
    )
    (root / "spoken.yaml").write_text("s1:\n  speech: Do the thing.\ns9:\n  speech: Ghost.\n")
    with pytest.raises(protocol.ProtocolError, match="s9"):
        protocol.load(tmp_path, "p")
