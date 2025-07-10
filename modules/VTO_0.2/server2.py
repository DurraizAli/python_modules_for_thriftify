import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, send_file, jsonify
import io

app = Flask(__name__)

def overlay_png(bg, fg, x, y):
    fg_h, fg_w, _ = fg.shape
    for i in range(fg_h):
        for j in range(fg_w):
            if y+i >= bg.shape[0] or x+j >= bg.shape[1] or x+j < 0 or y+i < 0:
                continue
            alpha = fg[i, j, 3] / 255.0
            if alpha > 0:
                bg[y+i, x+j] = (1 - alpha) * bg[y+i, x+j] + alpha * fg[i, j, :3]
    return bg

def tryon_top(imgPerson, imgCloth, result, mp_pose):
    h, w, _ = imgPerson.shape
    lm = result.pose_landmarks.landmark

    def get_point(p):
        return int(lm[p].x * w), int(lm[p].y * h)

    l_shldr = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shldr = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
    r_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)
    neck = get_point(mp_pose.PoseLandmark.NOSE)

    shoulder_center = ((l_shldr[0] + r_shldr[0]) // 2, (l_shldr[1] + r_shldr[1]) // 2)
    hip_center = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)

    shirt_width = int(np.linalg.norm(np.array(r_shldr) - np.array(l_shldr)) * 1.6)
    shirt_height = int((l_hip[1] - neck[1]) * 1)

    resizedShirt = cv2.resize(imgCloth, (shirt_width, shirt_height))
    top_left_x = shoulder_center[0] - shirt_width // 2
    top_left_y = neck[1] - int(shirt_height * -0.12)

    final_img = overlay_png(imgPerson.copy(), resizedShirt, top_left_x, top_left_y)
    return final_img

def tryon_bottom(imgPerson, imgCloth, result, mp_pose):
    h, w, _ = imgPerson.shape
    lm = result.pose_landmarks.landmark

    def get_point(p):
        return int(lm[p].x * w), int(lm[p].y * h)

    l_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
    r_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)
    l_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE)
    r_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)

    hip_center = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
    ankle_center = ((l_ankle[0] + r_ankle[0]) // 2, (l_ankle[1] + r_ankle[1]) // 2)

    pants_width = int(np.linalg.norm(np.array(r_hip) - np.array(l_hip)) * 1.2)
    pants_height = int((ankle_center[1] - hip_center[1]) * 1.05)

    resizedPants = cv2.resize(imgCloth, (pants_width, pants_height))
    top_left_x = hip_center[0] - pants_width // 2
    top_left_y = hip_center[1]

    final_img = overlay_png(imgPerson.copy(), resizedPants, top_left_x, top_left_y)
    return final_img

def tryon_full(imgPerson, imgCloth, result, mp_pose):
    h, w, _ = imgPerson.shape
    lm = result.pose_landmarks.landmark

    def get_point(p):
        return int(lm[p].x * w), int(lm[p].y * h)

    l_shldr = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER)
    r_shldr = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER)
    l_ankle = get_point(mp_pose.PoseLandmark.LEFT_ANKLE)
    r_ankle = get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)
    l_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP)
    r_hip = get_point(mp_pose.PoseLandmark.RIGHT_HIP)
    neck = get_point(mp_pose.PoseLandmark.NOSE)

    # Use the higher of the two shoulders for the anchor
    anchor_y = min(l_shldr[1], r_shldr[1]) - int(0.05 * h)  # Slightly above shoulders

    # For width, use distance between shoulders and add more width for chest/waist
    shoulder_width = np.linalg.norm(np.array(r_shldr) - np.array(l_shldr))
    hip_width = np.linalg.norm(np.array(r_hip) - np.array(l_hip))
    # belly_width = hip_width * 2
    dress_width = int(max(shoulder_width *1.7, hip_width * 2.1)*1.25)  # Increase multiplier for more natural fit

    # For height, from anchor to average ankle y
    ankle_center_y = (l_ankle[1] + r_ankle[1]) // 2
    dress_height = int((ankle_center_y - anchor_y) * 1.15) # Increase multiplier for more natural fit

    # Resize and position
    resizedDress = cv2.resize(imgCloth, (dress_width, dress_height))
    shoulder_center_x = (l_shldr[0] + r_shldr[0]) // 2
    top_left_x = shoulder_center_x - dress_width // 2
    top_left_y = anchor_y

    final_img = overlay_png(imgPerson.copy(), resizedDress, top_left_x, top_left_y)
    return final_img

@app.route('/try-on', methods=['POST'])
def try_on():
    if 'user' not in request.files or 'product' not in request.files:
        return jsonify({'error': 'Missing files'}), 400

    # Read images from request
    user_file = request.files['user']
    product_file = request.files['product']

    # Get type from form-data (default to 'top')
    cloth_type = request.form.get('type', 'top').lower()

    # Convert to OpenCV images
    user_bytes = np.frombuffer(user_file.read(), np.uint8)
    imgPerson = cv2.imdecode(user_bytes, cv2.IMREAD_COLOR)
    product_bytes = np.frombuffer(product_file.read(), np.uint8)
    imgCloth = cv2.imdecode(product_bytes, cv2.IMREAD_UNCHANGED)  # With alpha

    if imgPerson is None or imgCloth is None:
        return jsonify({'error': 'Error loading images'}), 400

    # Pose estimation
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True)
    imgRGB = cv2.cvtColor(imgPerson, cv2.COLOR_BGR2RGB)
    result = pose.process(imgRGB)

    if not result.pose_landmarks:
        return jsonify({'error': 'No pose detected'}), 400

    # Choose try-on function based on type
    if cloth_type == 'top':
        final_img = tryon_top(imgPerson, imgCloth, result, mp_pose)
    elif cloth_type == 'bottom':
        final_img = tryon_bottom(imgPerson, imgCloth, result, mp_pose)
    elif cloth_type == 'full':
        final_img = tryon_full(imgPerson, imgCloth, result, mp_pose)
    else:
        return jsonify({'error': 'Unknown type'}), 400

    # Encode result as PNG
    _, buffer = cv2.imencode('.png', final_img)
    io_buf = io.BytesIO(buffer)

    return send_file(io_buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)