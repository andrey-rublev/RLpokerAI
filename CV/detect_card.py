import cv2
import numpy as np

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

image_path = r"C:\Users\nikhi\Downloads\RLpoker\RLpokerAI\Data\tests\Test_Subject.jpg"
image = cv2.imread(image_path)
original = image.copy()

blur = cv2.GaussianBlur(image, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

warped = None
for c in contours:
    area = cv2.contourArea(c)
    if area > 1000:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = np.float32(approx)
            rect = order_points(pts.reshape(4, 2))
            
            widthA = np.sqrt(((rect[2][0] - rect[3][0]) ** 2) + ((rect[2][1] - rect[3][1]) ** 2))
            widthB = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((rect[1][0] - rect[2][0]) ** 2) + ((rect[1][1] - rect[2][1]) ** 2))
            heightB = np.sqrt(((rect[0][0] - rect[3][0]) ** 2) + ((rect[0][1] - rect[3][1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))
            
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))
            break

if warped is not None:
    warped_copy = warped.copy()
    warped_blur = cv2.GaussianBlur(warped, (5, 5), 0)
    warped_edges = cv2.Canny(warped_blur, 50, 150)
    warped_contours, _ = cv2.findContours(warped_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in warped_contours:
        area = cv2.contourArea(c)
        if area > 100:
            cv2.drawContours(warped_copy, [c], -1, (0, 255, 0), 2)
    
    # Extract top-left corner for rank and suit
    h, w = warped.shape[:2]
    corner_width = int(w * 0.25)  # Use 25% of card width
    corner_height = int(h * 0.25)  # Use 25% of card height
    top_left_corner = warped[0:corner_height, 0:corner_width]
    corner_edges = warped_edges[0:corner_height, 0:corner_width]

    # Create a white background and draw the outline (contours) on it
    outline_img = np.ones_like(top_left_corner) * 255
    corner_cnts, _ = cv2.findContours(corner_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cc in corner_cnts:
        if cv2.contourArea(cc) > 10:
            # draw contour in black on the white background
            cv2.drawContours(outline_img, [cc], -1, (0, 0, 0), 2)

    # Resize outline for better visibility (2x larger)
    outline_large = cv2.resize(outline_img, (corner_width * 2, corner_height * 2))

    cv2.imshow("Original Image", image)
    cv2.imshow("Straightened Card", warped)
    cv2.imshow("Contours on Straightened Card", warped_copy)
    cv2.imshow("Top-Left Corner (Outline)", outline_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()