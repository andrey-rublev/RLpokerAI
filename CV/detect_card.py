import cv2

image_path = r"C:\Users\nikhi\Downloads\RLpoker\RLpokerAI\Data\tests\Test_Subject.jpg"
image = cv2.imread(image_path)
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    area = cv2.contourArea(c)
    if area > 1000:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
cv2.imshow("Detected Card", image)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()