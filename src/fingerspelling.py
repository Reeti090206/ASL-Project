"""
Fingerspelling Word Builder Module.
Assembles single-letter sign language predictions into words and sentences.
Provides debouncing, hold-to-confirm, gesture-release detection, repeated-letter support,
and manual editing controls (Add, Undo, Space, Clear, Finish).
"""

from typing import List, Dict, Any, Optional


class FingerspellingManager:
    """
    State manager for sign language fingerspelling.
    Converts a stream of per-frame letter predictions into spelled words and sentences.
    """

    def __init__(
        self,
        min_hold_frames: int = 10,
        min_confidence: float = 0.50,
        auto_commit: bool = True,
    ):
        """
        Args:
            min_hold_frames: Consecutive frames required to auto-commit a letter.
            min_confidence: Minimum model confidence required to consider a prediction.
            auto_commit: If True, commits letter after holding steadily for min_hold_frames.
                         If False, letters must be confirmed manually via confirm_letter().
        """
        self.min_hold_frames = max(1, min_hold_frames)
        self.min_confidence = min_confidence
        self.auto_commit = auto_commit

        self.current_word: str = ""
        self.words_history: List[str] = []

        # Real-time state
        self.candidate_letter: Optional[str] = None
        self.candidate_confidence: Optional[float] = None
        self.hold_count: int = 0
        self.last_committed_letter: Optional[str] = None
        self.gesture_released: bool = True  # True if hand was removed or changed

    def process_frame_prediction(
        self,
        detected_letter: Optional[str],
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Processes a single-frame prediction from the alphabet classifier.

        Returns:
            Dict containing:
                - candidate_letter: currently held letter (or None)
                - confidence: model confidence
                - hold_progress: float 0.0 to 1.0 towards auto-commit
                - just_committed: letter that was auto-committed on this frame (or None)
                - current_word: current accumulated word string
                - transcript: full sentence so far
                - gesture_released: whether gesture release was satisfied
        """
        just_committed = None

        if not detected_letter:
            # Hand removed or no sign recognized -> resets hold and marks gesture as released
            self.candidate_letter = None
            self.candidate_confidence = None
            self.hold_count = 0
            self.gesture_released = True
            return self._build_status(just_committed=None)

        # Check confidence threshold if confidence is provided
        if confidence is not None and confidence < self.min_confidence:
            self.hold_count = max(0, self.hold_count - 1)
            return self._build_status(just_committed=None)

        # If detected letter is different from the last committed letter, gesture has changed
        if detected_letter != self.last_committed_letter:
            self.gesture_released = True

        if detected_letter == self.candidate_letter:
            self.hold_count += 1
            self.candidate_confidence = confidence
        else:
            self.candidate_letter = detected_letter
            self.candidate_confidence = confidence
            self.hold_count = 1

        # Check if we should auto-commit
        if (
            self.auto_commit
            and self.gesture_released
            and self.hold_count >= self.min_hold_frames
        ):
            self.confirm_letter(self.candidate_letter)
            just_committed = self.candidate_letter
            self.last_committed_letter = self.candidate_letter
            self.gesture_released = False  # Lock until hand releases or changes
            self.hold_count = 0

        return self._build_status(just_committed=just_committed)

    def confirm_letter(self, letter: Optional[str] = None) -> bool:
        """
        Appends a letter to the current word.
        If letter is not provided, commits candidate_letter if present.
        Supports repeating letters explicitly (e.g. LL in HELLO).
        """
        target = (letter or self.candidate_letter or "").strip().upper()
        if not target:
            return False

        self.current_word += target
        self.last_committed_letter = target
        self.hold_count = 0
        return True

    def undo_letter(self) -> Optional[str]:
        """Removes the last letter from the current word."""
        if not self.current_word:
            return None
        removed = self.current_word[-1]
        self.current_word = self.current_word[:-1]
        self.gesture_released = True
        return removed

    def add_space(self) -> None:
        """Finishes the current word and appends it to history."""
        self.finish_word()

    def finish_word(self) -> Optional[str]:
        """Finalizes the current word, adding it to words history."""
        word = self.current_word.strip()
        if word:
            self.words_history.append(word)
            self.current_word = ""
            self.last_committed_letter = None
            self.gesture_released = True
            return word
        return None

    def clear_current_word(self) -> None:
        """Clears only the word currently being spelled."""
        self.current_word = ""
        self.last_committed_letter = None
        self.gesture_released = True
        self.hold_count = 0

    def clear_all(self) -> None:
        """Resets both current word and words history."""
        self.current_word = ""
        self.words_history = []
        self.candidate_letter = None
        self.candidate_confidence = None
        self.hold_count = 0
        self.last_committed_letter = None
        self.gesture_released = True

    def get_transcript(self) -> str:
        """Returns the full sentence composed of completed words + current word."""
        parts = list(self.words_history)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def _build_status(self, just_committed: Optional[str]) -> Dict[str, Any]:
        progress = min(1.0, float(self.hold_count) / float(self.min_hold_frames))
        return {
            "candidate_letter": self.candidate_letter,
            "confidence": self.candidate_confidence,
            "hold_count": self.hold_count,
            "hold_progress": progress,
            "just_committed": just_committed,
            "current_word": self.current_word,
            "words_history": list(self.words_history),
            "transcript": self.get_transcript(),
            "gesture_released": self.gesture_released,
        }
