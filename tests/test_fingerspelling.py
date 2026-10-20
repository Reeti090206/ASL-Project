"""
Unit tests for FingerspellingManager.
Validates debouncing, gesture release detection, repeated-letter logic, and word assembly.
"""

import pytest
from src.fingerspelling import FingerspellingManager


def test_initial_state():
    mgr = FingerspellingManager()
    assert mgr.current_word == ""
    assert mgr.words_history == []
    assert mgr.get_transcript() == ""


def test_manual_letter_confirmation():
    mgr = FingerspellingManager(auto_commit=False)
    mgr.confirm_letter("H")
    mgr.confirm_letter("I")
    assert mgr.current_word == "HI"
    assert mgr.get_transcript() == "HI"

    # Finish word
    finished = mgr.finish_word()
    assert finished == "HI"
    assert mgr.current_word == ""
    assert mgr.words_history == ["HI"]
    assert mgr.get_transcript() == "HI"


def test_repeated_letters_support():
    mgr = FingerspellingManager(auto_commit=False)
    mgr.confirm_letter("H")
    mgr.confirm_letter("E")
    mgr.confirm_letter("L")
    mgr.confirm_letter("L")
    mgr.confirm_letter("O")
    assert mgr.current_word == "HELLO"


def test_undo_and_clear():
    mgr = FingerspellingManager(auto_commit=False)
    mgr.confirm_letter("C")
    mgr.confirm_letter("A")
    mgr.confirm_letter("T")
    mgr.confirm_letter("S")
    assert mgr.current_word == "CATS"

    removed = mgr.undo_letter()
    assert removed == "S"
    assert mgr.current_word == "CAT"

    mgr.clear_current_word()
    assert mgr.current_word == ""


def test_space_and_sentence_building():
    mgr = FingerspellingManager(auto_commit=False)
    mgr.confirm_letter("H")
    mgr.confirm_letter("I")
    mgr.add_space()

    mgr.confirm_letter("A")
    mgr.confirm_letter("L")
    mgr.confirm_letter("L")
    assert mgr.get_transcript() == "HI ALL"

    mgr.clear_all()
    assert mgr.get_transcript() == ""
    assert mgr.words_history == []
    assert mgr.current_word == ""


def test_hold_debouncing_and_gesture_release():
    # min_hold_frames = 5
    mgr = FingerspellingManager(min_hold_frames=5, auto_commit=True)

    # Send 'A' for 4 frames (not yet committed)
    for _ in range(4):
        status = mgr.process_frame_prediction("A", confidence=0.9)
        assert status["just_committed"] is None
        assert mgr.current_word == ""

    # 5th frame commits 'A'
    status = mgr.process_frame_prediction("A", confidence=0.9)
    assert status["just_committed"] == "A"
    assert mgr.current_word == "A"

    # Holding 'A' for 10 more frames must NOT commit additional 'A's!
    for _ in range(10):
        status = mgr.process_frame_prediction("A", confidence=0.9)
        assert status["just_committed"] is None
        assert mgr.current_word == "A"

    # Now release hand (None for 1 frame)
    status = mgr.process_frame_prediction(None)
    assert status["gesture_released"] is True

    # Now show 'A' again for 5 frames -> should commit second 'A'
    for _ in range(4):
        mgr.process_frame_prediction("A", confidence=0.9)
    status = mgr.process_frame_prediction("A", confidence=0.9)
    assert status["just_committed"] == "A"
    assert mgr.current_word == "AA"
