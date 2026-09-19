"""Loading a protocol, and what the loader refuses.

The loader is the only thing standing between a malformed file and an
assistant confidently reading nonsense aloud, so what it rejects matters as
much as what it accepts.
"""

from pathlib import Path

import pytest

from wetlab import protocol

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


@pytest.fixture
def hand():
    return protocol.load(CORPUS, "handwritten")


def write(root: Path, parsed: str, *, citation: str = "test", spoken: str | None = None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "parsed.yaml").write_text(parsed)
    if citation is not None:
        (root / "CITATION").write_text(citation)
    if spoken is not None:
        (root / "spoken.yaml").write_text(spoken)
    return root


ONE_STEP = "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: 'Add 25 microlitres.'\n"


def test_it_loads_a_protocol(hand):
    assert hand.id == "handwritten"
    assert hand.citation
    step = hand.steps[0]
    assert step.index == 0
    assert step.number == 1
    assert step.text


def test_a_step_carries_only_id_index_text_and_timers(hand):
    step = hand.steps[0]
    assert set(vars(step)) == {"id", "index", "text", "timers"}


def test_timers_are_loaded_with_their_label_and_length(hand):
    timer = hand.steps[0].timers[0]
    assert timer.seconds == 30
    assert timer.label


def test_step_looks_up_by_the_number_a_person_says(hand):
    assert hand.step(1).id == "s1"
    assert hand.step(2).id == "s2"


@pytest.mark.parametrize("number", [0, -1, 99])
def test_step_returns_none_outside_the_protocol(hand, number):
    assert hand.step(number) is None


def test_a_missing_citation_is_refused(tmp_path):
    """A protocol without a source is one nobody can check against anything."""
    root = write(tmp_path / "p", ONE_STEP, citation="   ")
    with pytest.raises(protocol.ProtocolError, match="CITATION"):
        protocol.load(tmp_path, "p")


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(protocol.ProtocolError, match="parsed.yaml"):
        protocol.load(tmp_path, "nothing")


def test_a_protocol_with_no_steps_is_refused(tmp_path):
    write(tmp_path / "p", "id: p\ntitle: t\nsteps: []\n")
    with pytest.raises(protocol.ProtocolError, match="no steps"):
        protocol.load(tmp_path, "p")


def test_a_step_with_no_text_is_refused(tmp_path):
    """Silence read aloud is indistinguishable from a dead worker."""
    write(tmp_path / "p", "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: '  '\n")
    with pytest.raises(protocol.ProtocolError, match="no text"):
        protocol.load(tmp_path, "p")


def test_duplicate_step_ids_are_refused(tmp_path):
    """Ids key the read-aloud record and the held audio, so two steps sharing
    one would make the second inherit the first's history."""
    write(
        tmp_path / "p",
        "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: 'One.'\n  - id: s1\n    text: 'Two.'\n",
    )
    with pytest.raises(protocol.ProtocolError, match="duplicate"):
        protocol.load(tmp_path, "p")


def test_the_prepared_form_is_what_gets_spoken(tmp_path):
    write(
        tmp_path / "p",
        ONE_STEP,
        spoken="s1:\n  speech: 'Add twenty five microlitres.'\n  terms: [volume, much, 25]\n",
    )
    loaded = protocol.load(tmp_path, "p")
    assert loaded.speech_for(loaded.steps[0]) == "Add twenty five microlitres."
    assert set(loaded.terms_for(loaded.steps[0])) == {"volume", "much", "25"}


def test_an_unprepared_protocol_will_not_load_for_reading(tmp_path):
    """There is no fallback any more.

    A rule table used to render a step with no prepared form, and it was worse
    at the job in exactly the cases that matter: 20-30 mins came out as
    negative thirty, and a slash was read as the word "slash". Speaking a step
    badly is not better than refusing to, because the listener cannot see the
    text to know it went wrong. Failing at load is better than failing on the
    first step, which would be halfway through a demo.
    """
    write(tmp_path / "p", "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: 'Add 25 µl.'\n")
    with pytest.raises(protocol.ProtocolError, match="prepare_protocol"):
        protocol.load(tmp_path, "p")


def test_an_empty_prepared_entry_counts_as_missing(tmp_path):
    write(tmp_path / "p", ONE_STEP, spoken="s1:\n  speech: ''\n")
    with pytest.raises(protocol.ProtocolError, match="prepared"):
        protocol.load(tmp_path, "p")


def test_the_build_scripts_can_still_load_an_unprepared_protocol(tmp_path):
    """Ingesting and preparing both have to read a protocol before its spoken
    form exists, so they opt out of the requirement rather than work around
    it."""
    write(tmp_path / "p", "id: p\ntitle: t\nsteps:\n  - id: s1\n    text: 'Add 25 µl.'\n")
    loaded = protocol.load(tmp_path, "p", require_spoken=False)
    assert loaded.steps[0].text == "Add 25 µl."
    assert loaded.spoken == {}
    with pytest.raises(protocol.ProtocolError):
        loaded.speech_for(loaded.steps[0])


# --- what the picker is offered ------------------------------------------------


def test_the_corpus_lists_every_protocol_in_it():
    listed = protocol.available(CORPUS)
    assert {p.id for p in listed} >= {"neb_q5_m0492", "addgene_transformation", "handwritten"}
    assert all(p.ready for p in listed), "every committed protocol should be prepared"
    assert all(p.citation for p in listed), "a protocol cannot be offered without a source"


def test_the_listed_id_is_the_directory_name_because_that_is_what_load_takes(tmp_path):
    """parsed.yaml carries an id of its own and is free to disagree with the
    directory it sits in. The picker sends this back and the agent hands it
    straight to `load`, which resolves a directory."""
    write(tmp_path / "on_disk", "id: elsewhere\ntitle: t\nsteps:\n  - id: s1\n    text: 'x'\n")
    listed = protocol.available(tmp_path)
    assert [p.id for p in listed] == ["on_disk"]
    assert protocol.load(tmp_path, listed[0].id, require_spoken=False).id == "elsewhere"


def test_a_protocol_with_no_prepared_form_is_listed_and_marked_not_ready(tmp_path):
    """Shown and refused rather than hidden. "This exists and has not been
    prepared" is more use to look at than a protocol that is silently absent."""
    write(tmp_path / "raw", ONE_STEP)
    write(tmp_path / "done", ONE_STEP, spoken="s1:\n  speech: 'Add twenty five microlitres.'\n")
    assert {p.id: p.ready for p in protocol.available(tmp_path)} == {"done": True, "raw": False}


def test_one_malformed_directory_does_not_hide_the_others(tmp_path):
    """The listing feeds the picker. One half-ingested directory must not be
    able to stop the rest of the corpus being offered."""
    write(tmp_path / "good", ONE_STEP, spoken="s1:\n  speech: 'Add twenty five.'\n")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "parsed.yaml").write_text("steps: [[[")
    write(tmp_path / "uncited", ONE_STEP, citation=None)
    assert [p.id for p in protocol.available(tmp_path)] == ["good"]


def test_a_corpus_that_is_not_there_lists_nothing_rather_than_raising(tmp_path):
    assert protocol.available(tmp_path / "nope") == ()
