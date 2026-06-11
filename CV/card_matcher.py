"""
Card identification via template matching.

Loads mean overlay templates for all 52 cards (built by build_templates.py),
then compares an unknown card's corner overlay using normalized cross-correlation.

Usage:
    from card_matcher import CardMatcher

    matcher = CardMatcher()

    # From a file path:
    label, confidence, all_matches = matcher.identify("unknown_card.jpg")
    print(f"{label}  ({confidence:.3f})")

    # From a numpy BGR array:
    label, confidence, all_matches = matcher.identify(image_array)

    # all_matches is a list of (label, score) sorted best-first, score in [-1, 1].
"""
import os
import sys
import cv2
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_card import detect_and_get_top_left, detect_card_from_image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE, "Data", "templates", "card_overlays")

ALL_CARDS = [
    f"{r}{s}"
    for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    for s in ["S", "H", "D", "C"]
]


class CardMatcher:
    """Match an unknown card image against all 52 card templates."""

    def __init__(self, templates_dir: str = TEMPLATES_DIR):
        self.templates_dir = templates_dir
        self.templates: dict[str, np.ndarray] = {}
        self._load_templates()

    def _load_templates(self) -> None:
        for card in ALL_CARDS:
            path = os.path.join(self.templates_dir, f"{card}.npy")
            if os.path.exists(path):
                self.templates[card] = np.load(path).astype(np.float32)

        if not self.templates:
            raise FileNotFoundError(
                f"No templates found in:\n  {self.templates_dir}\n"
                "Run build_templates.py first to generate them."
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        """Normalized cross-correlation, returns value in [-1, 1]."""
        a = a.flatten().astype(np.float64)
        b = b.flatten().astype(np.float64)
        a -= a.mean()
        b -= b.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < 1e-8:
            return 0.0
        return float(np.dot(a, b) / denom)

    def match_overlay(self, overlay: np.ndarray) -> list[tuple[str, float]]:
        """
        Compare a 64×96 grayscale overlay against all loaded templates.

        Parameters
        ----------
        overlay : np.ndarray
            64×96 uint8 grayscale image (from detect_and_get_top_left).

        Returns
        -------
        list of (label, score) sorted by score descending.
        Score is normalized cross-correlation in [-1, 1]; higher = better match.
        """
        query = overlay.astype(np.float32)
        results = [
            (label, self._ncc(query, tmpl))
            for label, tmpl in self.templates.items()
        ]
        results.sort(key=lambda x: -x[1])
        return results

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def identify(
        self,
        image_or_path,
        debug: bool = False,
    ) -> tuple[Optional[str], float, list[tuple[str, float]]]:
        """
        Identify a card from an image file path or BGR numpy array.

        Returns
        -------
        (best_label, confidence, all_matches)
            best_label  : str like 'AS', '10C', 'KH', or None on failure.
            confidence  : float in [-1, 1], NCC score of best match.
            all_matches : full sorted list of (label, score) for all 52 cards.
        """
        if isinstance(image_or_path, str):
            result = detect_and_get_top_left(image_or_path, debug=debug)
        else:
            result = detect_card_from_image(image_or_path, debug=debug)

        if result is None:
            return None, 0.0, []

        overlay = result["overlay"]
        matches = self.match_overlay(overlay)

        if not matches:
            return None, 0.0, []

        best_label, best_score = matches[0]
        return best_label, best_score, matches

    def identify_overlay(
        self,
        overlay: np.ndarray,
    ) -> tuple[Optional[str], float, list[tuple[str, float]]]:
        """
        Identify directly from a pre-extracted 64×96 overlay.
        Useful when you already have the overlay from detect_and_get_top_left.
        """
        matches = self.match_overlay(overlay)
        if not matches:
            return None, 0.0, []
        best_label, best_score = matches[0]
        return best_label, best_score, matches
