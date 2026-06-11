"""
Poker board analyzer: detect and identify all cards in a single scene image.

Finds every card-shaped contour, straightens each one, extracts the corner
overlay, and matches it against the 52-card template database.  Then clusters
the identified cards into community cards (center of table) and player hands
(edges).

Usage:
    from poker_board import PokerBoard

    board = PokerBoard()
    result = board.analyze("poker_table.jpg")

    for card in result["community_cards"]:
        print(card.label, card.confidence)

    for i, hand in enumerate(result["player_hands"]):
        print(f"Player {i+1}:", [c.label for c in hand])

    # Save annotated image:
    cv2.imwrite("annotated.jpg", result["annotated_image"])
"""
import os
import sys
import cv2
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_card import straighten_card_perspective
from card_matcher import CardMatcher

TARGET_W, TARGET_H = 64, 96  # must match detect_card.py


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class DetectedCard:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in original image coords
    center: tuple[int, int]            # (cx, cy)
    warped: np.ndarray = field(repr=False)   # straightened BGR crop


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _extract_overlay(warped: np.ndarray) -> np.ndarray:
    """Extract binarized 64×96 corner overlay from a straightened card image."""
    h, w = warped.shape[:2]
    cw = max(1, int(w * 0.25))
    ch = max(1, int(h * 0.35))
    corner = warped[0:ch, 0:cw]

    gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    return cv2.resize(bw, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)


def _find_card_contours(
    image: np.ndarray,
    min_card_fraction: float = 0.003,
    max_card_fraction: float = 0.50,
) -> list:
    """
    Find all card-shaped contours in image.

    Cards are identified as large, roughly rectangular contours with an
    aspect ratio consistent with a playing card (0.4 – 2.0 to allow for
    partial occlusion and tilted cards).
    """
    img_area = image.shape[0] * image.shape[1]
    min_area = img_area * min_card_fraction
    max_area = img_area * max_card_fraction

    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)
        if 0.4 <= aspect <= 2.0:
            candidates.append(c)

    # Largest first
    candidates.sort(key=cv2.contourArea, reverse=True)
    return candidates


def _iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter)


def _nms(contours: list, iou_thresh: float = 0.4) -> list:
    """Non-maximum suppression: remove heavily overlapping contours."""
    if not contours:
        return []
    boxes = [cv2.boundingRect(c) for c in contours]
    keep = []
    suppressed = set()
    for i in range(len(contours)):
        if i in suppressed:
            continue
        keep.append(contours[i])
        for j in range(i + 1, len(contours)):
            if j not in suppressed and _iou(boxes[i], boxes[j]) > iou_thresh:
                suppressed.add(j)
    return keep


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PokerBoard:
    """
    Detect and identify all playing cards in a poker table scene.

    Parameters
    ----------
    templates_dir : optional path to card template directory.
                    Defaults to Data/templates/card_overlays/.
    min_confidence : float
        Cards with NCC score below this threshold are discarded.
    """

    def __init__(
        self,
        templates_dir: str = None,
        min_confidence: float = 0.10,
    ):
        kwargs = {"templates_dir": templates_dir} if templates_dir else {}
        self.matcher = CardMatcher(**kwargs)
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_cards(self, image: np.ndarray) -> list[DetectedCard]:
        """
        Locate and identify every card in a BGR scene image.

        Returns a list of DetectedCard objects, deduplicated by label
        (highest-confidence detection kept when the same card is seen twice).
        """
        contours = _find_card_contours(image)
        contours = _nms(contours)

        raw: list[DetectedCard] = []

        for contour in contours:
            warped = straighten_card_perspective(image, contour)
            if warped is None or warped.size == 0:
                continue

            h, w = warped.shape[:2]
            if w > h:
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h, w = warped.shape[:2]

            # Discard cards that are too small to read reliably
            if w < 30 or h < 50:
                continue

            overlay = _extract_overlay(warped)
            label, confidence, _ = self.matcher.identify_overlay(overlay)

            if label is None or confidence < self.min_confidence:
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            raw.append(DetectedCard(
                label=label,
                confidence=confidence,
                bbox=(x, y, bw, bh),
                center=(x + bw // 2, y + bh // 2),
                warped=warped,
            ))

        # Keep only the highest-confidence detection per label
        best: dict[str, DetectedCard] = {}
        for card in raw:
            if card.label not in best or card.confidence > best[card.label].confidence:
                best[card.label] = card

        return list(best.values())

    # ------------------------------------------------------------------
    # Layout analysis
    # ------------------------------------------------------------------

    def _split_community_players(
        self,
        cards: list[DetectedCard],
        image_shape: tuple,
        community_count: int,
    ) -> tuple[list[DetectedCard], list[DetectedCard]]:
        """
        Split cards into community (center of image) and player cards (edges).

        Community cards are the `community_count` cards whose vertical center
        is closest to the image's vertical midpoint.
        """
        if len(cards) <= community_count:
            community = sorted(cards, key=lambda c: c.center[0])
            return community, []

        img_h = image_shape[0]
        mid_y = img_h / 2.0
        by_dist = sorted(cards, key=lambda c: abs(c.center[1] - mid_y))

        community = sorted(by_dist[:community_count], key=lambda c: c.center[0])
        players = by_dist[community_count:]
        return community, players

    def _group_hands(self, cards: list[DetectedCard]) -> list[list[DetectedCard]]:
        """
        Group player cards into hands of 2 by proximity (x-axis clustering).

        Cards are sorted left-to-right and paired greedily; a new hand starts
        whenever the gap between adjacent cards exceeds a threshold.
        """
        if not cards:
            return []

        sorted_cards = sorted(cards, key=lambda c: c.center[0])

        if len(sorted_cards) <= 2:
            return [sorted_cards]

        # Compute gaps between adjacent card centers
        gaps = [
            sorted_cards[i + 1].center[0] - sorted_cards[i].center[0]
            for i in range(len(sorted_cards) - 1)
        ]
        median_gap = float(np.median(gaps))
        split_thresh = median_gap * 2.0

        hands: list[list[DetectedCard]] = []
        current_hand: list[DetectedCard] = [sorted_cards[0]]

        for i, card in enumerate(sorted_cards[1:], start=1):
            gap = sorted_cards[i].center[0] - sorted_cards[i - 1].center[0]
            if gap > split_thresh:
                hands.append(current_hand)
                current_hand = [card]
            else:
                current_hand.append(card)

        hands.append(current_hand)
        return hands

    # ------------------------------------------------------------------
    # Annotation rendering
    # ------------------------------------------------------------------

    _COMMUNITY_COLOR = (50, 220, 50)
    _PLAYER_COLORS = [
        (255, 100, 30),
        (30, 100, 255),
        (30, 230, 220),
        (220, 30, 220),
        (220, 180, 30),
        (30, 180, 220),
    ]

    def _draw(
        self,
        image: np.ndarray,
        community: list[DetectedCard],
        player_hands: list[list[DetectedCard]],
    ) -> np.ndarray:
        out = image.copy()

        def _label_card(card: DetectedCard, text: str, color: tuple):
            x, y, w, h = card.bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale, thick = 0.65, 2
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            ty = max(th + 4, y - 6)
            cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 4), color, -1)
            cv2.putText(out, text, (x + 2, ty), font, scale, (255, 255, 255), thick)

        for card in community:
            _label_card(card, f"{card.label} ({card.confidence:.2f})", self._COMMUNITY_COLOR)

        for pi, hand in enumerate(player_hands):
            color = self._PLAYER_COLORS[pi % len(self._PLAYER_COLORS)]
            for card in hand:
                _label_card(card, f"P{pi+1}:{card.label} ({card.confidence:.2f})", color)

        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        image_or_path,
        community_count: int = 5,
    ) -> dict:
        """
        Analyze a poker board scene.

        Parameters
        ----------
        image_or_path : str or np.ndarray
            File path or BGR numpy array of the poker table.
        community_count : int
            How many community cards to expect (default 5 for Texas Hold'em).

        Returns
        -------
        dict with keys:
            "all_cards"        : list[DetectedCard]  – every identified card
            "community_cards"  : list[DetectedCard]  – board / community cards
            "player_hands"     : list[list[DetectedCard]]  – one list per player
            "annotated_image"  : np.ndarray  – BGR image with bounding boxes
        """
        if isinstance(image_or_path, str):
            image = cv2.imread(image_or_path)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image_or_path}")
        else:
            image = image_or_path.copy()

        cards = self.detect_cards(image)

        community, player_cards = self._split_community_players(
            cards, image.shape, community_count
        )
        player_hands = self._group_hands(player_cards)

        annotated = self._draw(image, community, player_hands)

        return {
            "all_cards": cards,
            "community_cards": community,
            "player_hands": player_hands,
            "annotated_image": annotated,
        }
