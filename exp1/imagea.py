import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("/home/tushara/Documents/Ashoka/Lab 4/Data/exp1/thickness_measurement.png", cv2.IMREAD_GRAYSCALE)

img_blur = cv2.medianBlur(img, 5)

circles = cv2.HoughCircles(
    img_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.2,           # accumulator resolution
    minDist=20,       # min spacing between circle centers
    param1=50,        # Canny high threshold
    param2=30,        # accumulator threshold (lower → more circles)
    minRadius=10,     # ignore very small circles
    maxRadius=0       # 0 = no upper bound
)

if circles is not None:
    circles = np.squeeze(circles) 

    cx, cy = np.median(circles[:,0]), np.median(circles[:,1])
    print(f"Estimated center: ({cx:.2f}, {cy:.2f})")

    radii_sorted = np.sort(circles[:,2].astype(float))
    groups = []
    tol = 3.0  
    current = [radii_sorted[0]]
    for r in radii_sorted[1:]:
        if abs(r - current[-1]) <= tol:
            current.append(r)
        else:
            groups.append(np.mean(current))
            current = [r]
    groups.append(np.mean(current))
    
    groups = np.sort(groups)
    
    if len(groups) >= 2:
        r1, r2 = groups[0], groups[1]
        print(f"First fringe radius: {r1:.2f} px")
        print(f"Second fringe radius: {r2:.2f} px")
        
        scale = 2.8 * (1024/960)  # µm per pixel
        print(f"Scale: {scale:.3f} µm/px")
        print(f"r1 = {r1*scale:.2f} µm, r2 = {r2*scale:.2f} µm")

    else:
        print("Not enough fringes detected.")
    
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in circles.astype(int):
        cv2.circle(output, (x, y), r, (0, 255, 0), 1)
    cv2.circle(output, (int(round(cx)), int(round(cy))), 3, (0, 0, 255), -1)

    plt.figure(figsize=(6,6))
    plt.imshow(output[...,::-1])
    plt.axis("off")
    plt.title("Detected fringes and center")
    plt.show()

else:
    print("No circles detected.")

