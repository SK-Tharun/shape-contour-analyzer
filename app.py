import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

#  PAGE CONFIG 
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title("🔍 Shape & Contour Analyzer")

st.markdown("""
Geometric shape detection using **strict mathematical rules**.
""")

#  SIDEBAR 
st.sidebar.header("⚙️ Controls")
min_area = st.sidebar.slider("Minimum Object Area", 100, 20000, 500)

#  IMAGE UPLOAD 
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

#  GEOMETRY HELPERS 
def angle_between(v1, v2):
    cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

def internal_angle(p1, p2, p3):
    return angle_between(p1 - p2, p3 - p2)

def is_parallel(p1, p2, p3, p4):
    v1 = p2 - p1
    v2 = p4 - p3
    ang = angle_between(v1, v2)
    return ang < 2 or abs(ang - 180) < 2

def side_lengths(pts):
    return [
        np.linalg.norm(pts[i] - pts[(i + 1) % len(pts)])
        for i in range(len(pts))
    ]

def opposite_angles_equal(angles, tol=6):
    return abs(angles[0] - angles[2]) < tol and abs(angles[1] - angles[3]) < tol

#  SHAPE CLASSIFIER 
def classify_shape(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return "Unknown"

    #  CIRCLE 
    circularity = 4 * np.pi * area / (peri * peri)
    if circularity > 0.85:
        return "Circle"

    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    v = len(approx)

    #  TRIANGLE 
    if v == 3:
        return "Triangle"

    #  QUADRILATERALS 
    if v == 4:
        pts = approx.reshape(4, 2)

        sides = side_lengths(pts)
        avg = np.mean(sides)
        equal_sides = max(sides) - min(sides) < 0.15 * avg

        angles = [internal_angle(pts[i-1], pts[i], pts[(i+1) % 4]) for i in range(4)]
        right_angles = all(88 <= a <= 92 for a in angles)
        opp_angles_eq = opposite_angles_equal(angles)

        parallel_pairs = 0
        if is_parallel(pts[0], pts[1], pts[2], pts[3]):
            parallel_pairs += 1
        if is_parallel(pts[1], pts[2], pts[3], pts[0]):
            parallel_pairs += 1

        #  STRICT ORDER 
        if equal_sides and right_angles:
            return "Square"

        if right_angles and parallel_pairs == 2:
            return "Rectangle"

        if equal_sides and opp_angles_eq:
            return "Rhombus"

        #  KITE 
        if (
            abs(sides[0] - sides[1]) < 0.15 * avg and
            abs(sides[2] - sides[3]) < 0.15 * avg and
            not opp_angles_eq
        ):
            return "Kite"

        if parallel_pairs == 2:
            return "Parallelogram"

        if parallel_pairs == 1:
            return "Trapezium"

        return "Quadrilateral"

    #  POLYGONS 
    names = {
        5: "Pentagon",
        6: "Hexagon",
        7: "Heptagon",
        8: "Octagon",
        9: "Nonagon"
    }
    return names.get(v, "Polygon")

# MAIN 
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    img_area = h * w
    output = img.copy()
    rows = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > 0.9 * img_area:
            continue

        peri = cv2.arcLength(cnt, True)
        shape = classify_shape(cnt)

        cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                output,
                shape,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),   # dark readable text
                2,
                cv2.LINE_AA
            )

        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        rows.append({
            "Shape": shape,
            "Vertices": 0 if shape == "Circle" else len(approx),
            "Area (px²)": round(area, 2),
            "Perimeter (px)": round(peri, 2)
        })


    df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("🧠 Detected Shapes")
        st.image(output, use_container_width=True)

    st.subheader(f"📊 Total Objects Detected: {len(df)}")
    st.dataframe(df, use_container_width=True)
