"""
Poker CV – CLI entry point.

Commands
--------
  build                           Build template database from dataset (run once).
  identify <image>                Identify a single card and show ranked matches.
  board    <image> [output.jpg]   Detect all cards on a poker table.

Examples
--------
  python main.py build
  python main.py identify Data/tests/Test_Subject.jpg
  python main.py board poker_table.jpg annotated_output.jpg
"""
import sys
import os
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build():
    from build_templates import build_templates
    print("Building card templates from dataset (this may take a few minutes)...\n")
    counts = build_templates(verbose=True)
    print(f"\nDone. {len(counts)}/52 templates built.")


def cmd_identify(image_path: str):
    from card_matcher import CardMatcher

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Identifying card in: {image_path}\n")
    matcher = CardMatcher()
    label, confidence, all_matches = matcher.identify(image_path)

    if label is None:
        print("No card detected in image.")
        return

    print(f"Best match:  {label}   (confidence: {confidence:+.4f})\n")
    print("Top 10 matches:")
    print(f"  {'Card':<6}  {'Score':>8}  {'Bar'}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*24}")
    for rank_label, score in all_matches[:10]:
        bar_len = int(max(0.0, score) * 24)
        bar = "█" * bar_len
        marker = " ◄" if rank_label == label else ""
        print(f"  {rank_label:<6}  {score:+.4f}   {bar}{marker}")


def cmd_board(image_path: str, output_path: str = None):
    from poker_board import PokerBoard

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing poker board: {image_path}\n")
    board = PokerBoard()
    result = board.analyze(image_path)

    all_cards = result["all_cards"]
    community = result["community_cards"]
    player_hands = result["player_hands"]

    print(f"Total cards detected: {len(all_cards)}")

    print("\nCommunity cards:")
    if community:
        for c in community:
            print(f"  {c.label:<4}  confidence: {c.confidence:+.3f}")
    else:
        print("  (none detected)")

    print("\nPlayer hands:")
    if player_hands:
        for i, hand in enumerate(player_hands):
            cards_str = "  ".join(
                f"{c.label}({c.confidence:+.2f})" for c in hand
            )
            print(f"  Player {i+1}:  {cards_str}")
    else:
        print("  (none detected)")

    annotated = result["annotated_image"]
    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\nAnnotated image saved to: {output_path}")
    else:
        h, w = annotated.shape[:2]
        scale = min(1.0, 1280 / max(h, w))
        display = cv2.resize(annotated, (int(w * scale), int(h * scale)))
        cv2.imshow("Poker Board Analysis", display)
        print("\n(Press any key in the image window to close.)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == "build":
        cmd_build()

    elif cmd == "identify":
        if len(sys.argv) < 3:
            print("Usage: python main.py identify <image_path>")
            sys.exit(1)
        cmd_identify(sys.argv[2])

    elif cmd == "board":
        if len(sys.argv) < 3:
            print("Usage: python main.py board <image_path> [output_path]")
            sys.exit(1)
        output = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_board(sys.argv[2], output)

    else:
        print(f"Unknown command: '{cmd}'\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
