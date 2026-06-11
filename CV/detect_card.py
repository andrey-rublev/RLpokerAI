import cv2
import numpy as np
import os
import glob

def straighten_card_perspective(image, contour, debug=False):
    """Straighten card using perspective transformation.
    Finds 4 corner points from contour and applies perspective correction."""
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if debug:
        print(f"Contour approximation has {len(approx)} vertices")
    
    # If we don't have exactly 4 points, use convex hull or minAreaRect
    if len(approx) != 4:
        if debug:
            print("Contour doesn't have 4 points, using convex hull")
        hull = cv2.convexHull(contour)
        # Get the 4 corners of the convex hull
        if len(hull) >= 4:
            # Use minAreaRect to get 4 corners
            rect = cv2.minAreaRect(hull)
            approx = cv2.boxPoints(rect)
            approx = np.int0(approx)
        else:
            if debug:
                print("Could not get 4 corners")
            return None
    
    # Get 4 corners
    pts = approx.reshape(4, 2).astype(np.float32)
    
    if debug:
        print(f"Corner points: {pts}")
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    pts_ordered = order_points_simple(pts)
    
    if debug:
        print(f"Ordered points: {pts_ordered}")
    
    # Get the width and height based on edge lengths
    width_top = np.linalg.norm(pts_ordered[1] - pts_ordered[0])
    width_bottom = np.linalg.norm(pts_ordered[2] - pts_ordered[3])
    height_left = np.linalg.norm(pts_ordered[3] - pts_ordered[0])
    height_right = np.linalg.norm(pts_ordered[2] - pts_ordered[1])
    
    card_width = int((width_top + width_bottom) / 2)
    card_height = int((height_left + height_right) / 2)
    
    if debug:
        print(f"Card dimensions: {card_width}x{card_height}")
    
    # Create destination rectangle
    dst = np.array([
        [0, 0],
        [card_width, 0],
        [card_width, card_height],
        [0, card_height]
    ], dtype="float32")
    
    # Apply perspective transformation
    M = cv2.getPerspectiveTransform(pts_ordered, dst)
    warped = cv2.warpPerspective(image, M, (card_width, card_height),
                                 flags=cv2.INTER_LINEAR)
    
    return warped

def order_points_simple(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left"""
    # Sort by y coordinate first to separate top and bottom
    sorted_y = pts[np.argsort(pts[:, 1])]
    
    # Top 2 points
    top = sorted_y[:2]
    # Bottom 2 points
    bottom = sorted_y[2:]
    
    # Sort top points by x
    tl, tr = top[np.argsort(top[:, 0])]
    # Sort bottom points by x
    bl, br = bottom[np.argsort(bottom[:, 0])]
    
    return np.array([tl, tr, br, bl], dtype=np.float32)

def order_points(pts):
    """Order 4 corner points to: top-left, top-right, bottom-right, bottom-left"""
    # Calculate center point
    center = pts.mean(axis=0)
    
    # Calculate angles from center to each point
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    
    # Sort points by angle
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    
    # Find top-left (point with minimum x+y sum)
    sums = sorted_pts[:, 0] + sorted_pts[:, 1]
    tl_idx = np.argmin(sums)
    
    # Reorder to start from top-left going clockwise
    ordered = np.roll(sorted_pts, -tl_idx, axis=0)
    
    return ordered.astype(np.float32)

def detect_and_get_top_left(image_path,
                          corner_frac: float = 0.25,
                          outline_scale: int = 2,
                          corner_area_thresh: int = 10,
                          target_size=(64, 96),
                          show: bool = False,
                          save_path: str = None,
                          debug: bool = False,
                          _image: np.ndarray = None):
    """Detect card in the image using curved edge detection, straighten it, and return the top-left outline crop.

    Uses edge detection and contour analysis to find the card's curved boundary (rounded corners).
    Returns a dict with keys:
      - warped: straightened card image (BGR)
      - top_left_corner: color crop of top-left (BGR)
      - corner_edges: edge map of the top-left crop (uint8)
      - outline: white background with black contour strokes (BGR)
      - outline_large: scaled-up version for display

    If save_path is provided, the outline_large will be saved to that path.
    If debug is True, prints contour info to help diagnose detection issues.
    """
    # If the image path doesn't exist, try to resolve common mistakes
    if image_path is not None and not os.path.exists(image_path):
        # Try same basename with common extensions in the Data images folder
        base = os.path.splitext(os.path.basename(image_path))[0]
        # common folders to search (relative to repo)
        search_dirs = [
            os.path.join(os.getcwd(), "Data", "Images", "Images"),
            os.path.join(os.getcwd(), "Data", "Images"),
            os.path.join(os.getcwd(), "Data")
        ]
        found = None
        for d in search_dirs:
            if not os.path.isdir(d):
                continue
            # look for files starting with base name
            pattern = os.path.join(d, base + ".*")
            matches = glob.glob(pattern)
            if matches:
                # prefer png/jpg/jpeg
                for ext in ('.png', '.jpg', '.jpeg'):
                    for m in matches:
                        if m.lower().endswith(ext):
                            found = m
                            break
                    if found:
                        break
                if not found:
                    found = matches[0]
                break
        if found:
            if debug:
                print(f"Image path not found; using '{found}' instead of '{image_path}'")
            image_path = found

    if _image is not None:
        image = _image
    else:
        image = cv2.imread(image_path)
        if image is None:
            parent = os.path.dirname(image_path) or os.getcwd()
            sample = []
            try:
                sample = [os.path.basename(p) for p in glob.glob(os.path.join(parent, '*'))][:20]
            except Exception:
                sample = []
            raise FileNotFoundError(f"Image not found: {image_path}. Nearby files: {sample}")
    original = image.copy()
    h, w = image.shape[:2]

    # Use multiple edge detection methods for robustness
    # 1. Canny edge detection
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    edges_canny = cv2.Canny(blur, 30, 100)
    
    # 2. Dilate and erode to connect nearby edges (helps with rounded corners)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_dilated = cv2.dilate(edges_canny, kernel, iterations=2)
    edges_processed = cv2.erode(edges_dilated, kernel, iterations=1)
    
    # Find contours on processed edges
    contours, _ = cv2.findContours(edges_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"Total contours found: {len(contours)}")

    # Find the largest contour (the card itself) - don't require it to be a perfect quadrilateral
    # since curved corners won't form a perfect 4-point approximation
    largest_contour = None
    largest_area = 0
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        
        if debug and area > 5000:  # Only print large contours
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            print(f"  Contour {i}: area={area:.0f}, perimeter={peri:.0f}, approx_sides={len(approx)}")
        
        # Prefer contours that are large and have reasonable aspect ratio (card-like)
        if area > largest_area and area > 5000:  # Minimum card size
            x, y, cw, ch = cv2.boundingRect(c)
            aspect_ratio = float(cw) / ch if ch > 0 else 0
            # Card aspect ratio is typically between 0.6 and 0.75 (portrait orientation varies)
            if 0.5 < aspect_ratio < 1.5:  # Allow some variation
                largest_contour = c
                largest_area = area
    
    if largest_contour is None:
        if debug:
            print("No suitable card contour found!")
        return None
    
    if debug:
        print(f"Using largest contour with area={largest_area:.0f}")
    
    # Straighten the card using perspective transformation based on detected corners
    warped = straighten_card_perspective(original, largest_contour, debug=debug)
    
    # Get dimensions of straightened card
    h, w = warped.shape[:2]

    # Ensure card is upright (portrait orientation: height > width)
    # If card is wider than tall, rotate 90 degrees
    h, w = warped.shape[:2]
    if w > h:
        if debug:
            print(f"Card is landscape ({w}x{h}), rotating to portrait...")
        warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        h, w = warped.shape[:2]

    warped_copy = warped.copy()
    warped_blur = cv2.GaussianBlur(warped, (5, 5), 0)
    warped_edges = cv2.Canny(warped_blur, 50, 150)
    warped_contours, _ = cv2.findContours(warped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in warped_contours:
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(warped_copy, [c], -1, (0, 255, 0), 2)

    # Extract top-left corner for rank and suit
    corner_width = int(w * corner_frac)
    corner_height = int(h * 0.35)  # Taller crop (35% of height instead of 25%)
    top_left_corner = warped[0:corner_height, 0:corner_width]
    corner_edges = warped_edges[0:corner_height, 0:corner_width]

    # Create a black and white overlay using binary thresholding
    gray = cv2.cvtColor(top_left_corner, cv2.COLOR_BGR2GRAY)
    # Apply blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Otsu's threshold for automatic thresholding - keeps background white, makes darker areas black
    _, bw_overlay = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bw_overlay = cv2.morphologyEx(bw_overlay, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw_overlay = cv2.morphologyEx(bw_overlay, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Convert to 3-channel for consistency (BGR format)
    bw_overlay_bgr = cv2.cvtColor(bw_overlay, cv2.COLOR_GRAY2BGR)

    # Resize overlay to a fixed target resolution so outputs are consistent
    # target_size is (width, height)
    target_w, target_h = target_size
    bw_overlay_resized = cv2.resize(bw_overlay, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    bw_overlay_resized_bgr = cv2.cvtColor(bw_overlay_resized, cv2.COLOR_GRAY2BGR)

    # Create a larger display image by scaling the fixed-size overlay
    overlay_large = cv2.resize(bw_overlay_resized_bgr, (target_w * outline_scale, target_h * outline_scale), 
                                interpolation=cv2.INTER_NEAREST)

    if save_path:
        cv2.imwrite(save_path, overlay_large)

    if show:
        # Helper function to resize image for display with max size constraint
        def resize_for_display(img, max_size=800):
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            return img
        
        cv2.imshow("Original Image", resize_for_display(image))
        cv2.imshow("Straightened Card", resize_for_display(warped))
        cv2.imshow("Contours on Straightened Card", resize_for_display(warped_copy))
        cv2.imshow("Top-Left Corner (B&W Overlay)", overlay_large)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "warped": warped,
        "top_left_corner": top_left_corner,
        "corner_edges": corner_edges,
        # resized overlay at fixed resolution (grayscale)
        "overlay": bw_overlay_resized,
        # BGR versions for display
        "overlay_bgr": bw_overlay_resized_bgr,
        "overlay_large": overlay_large,
    }


def detect_card_from_image(image: np.ndarray, **kwargs):
    """Convenience wrapper: run detect_and_get_top_left on a BGR numpy array."""
    return detect_and_get_top_left(None, _image=image, **kwargs)


if __name__ == "__main__":
    # simple CLI behavior remains for quick testing
    image_path = r"C:\Users\nikhi\Downloads\RLpoker\RLpokerAI\Data\Images\Images\2s1.jpg"
    res = detect_and_get_top_left(image_path, show=True, debug=True)
    if res is None:
        print("No card detected.")