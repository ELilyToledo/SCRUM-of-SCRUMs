import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk, Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import json
import os

# ---------------- Persistent Accounts ----------------
ACCOUNTS_FILE = "accounts.json"


# Functions and login algorithms from Bennet Ortiz
def load_accounts():
    if os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "r") as file:
            return json.load(file)
    else:
        # Default account if no file exists
        return {"admin": "password123"}

def save_accounts(accounts):
    with open(ACCOUNTS_FILE, "w") as file:
        json.dump(accounts, file)

accounts = load_accounts()

# ---------------- Global Lane Detection Variables ----------------
prev_left, prev_right, prev_center = None, None, None
smoothing_alpha = 0.1
last_vp = None
center_hist = deque(maxlen=30)

# ---------------- Lane Detection Functions ----------------
def update_hist(center_line):
    if center_line is not None:
        center_hist.append(center_line)

def predict_center(frame_h, poly_order=2):
    if len(center_hist) < 5:
        return None
    xs, ys = [], []
    for cl in center_hist:
        xs.extend([cl[0], cl[2]])
        ys.extend([cl[1], cl[3]])
    coeff = np.polyfit(ys, xs, poly_order)
    poly = np.poly1d(coeff)
    y_vals = np.linspace(int(0.5 * frame_h), frame_h, num=50)
    return [(int(poly(y)), int(y)) for y in y_vals]

def undistort(frame):
    h, w = frame.shape[:2]
    mtx = np.array([[w, 0, w / 2],
                    [0, w, h / 2],
                    [0, 0, 1]], dtype=np.float32)
    return cv2.undistort(frame, mtx, np.zeros((5, 1), np.float32), None, mtx)

def roi(img, verts):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verts, 255)
    return cv2.bitwise_and(img, mask)

def detect_lanes(frame):
    global prev_left, prev_right  # For lane history fallback
    h, w = frame.shape[:2]
    bot_off = int(0.1 * h)
    verts = np.array([[
        (0, h - bot_off),
        (int(0.1 * w), int(0.5 * h)),
        (int(0.9 * w), int(0.5 * h)),
        (w, h - bot_off)
    ]], dtype=np.int32)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    cropped = roi(edges, verts)
    # Adjusted HoughLinesP parameters: lower minLineLength and higher maxLineGap
    lines = cv2.HoughLinesP(cropped, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=250)

    left_lines, right_lines = [], []
    if lines is not None:
        for l in lines:
            for x1, y1, x2, y2 in l:
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if slope < -0.5:
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0.5:
                    right_lines.append([x1, y1, x2, y2])
    if lines is None or (not left_lines and not right_lines):
        return None, None, None, verts, False

    def avg_line(lns):
        if not lns:
            return None
        xs = [pt for line in lns for pt in (line[0], line[2])]
        ys = [pt for line in lns for pt in (line[1], line[3])]
        poly = np.polyfit(ys, xs, 1)
        s, i = poly
        yb, yt = h - bot_off, int(0.5 * h)
        return [int(s * yb + i), yb, int(s * yt + i), yt]

    left_avg = avg_line(left_lines)
    right_avg = avg_line(right_lines)
    default_left = [int(0.3 * w), h - bot_off, int(0.3 * w), int(0.5 * h)]
    default_right = [int(0.7 * w), h - bot_off, int(0.7 * w), int(0.5 * h)]

    # Use lane history to fill in gaps
    if left_avg is None:
        left_avg = prev_left if prev_left is not None else default_left
    if right_avg is None:
        right_avg = prev_right if prev_right is not None else default_right

    center_line_val = [(left_avg[0] + right_avg[0]) // 2,
                       (left_avg[1] + right_avg[1]) // 2,
                       (left_avg[2] + right_avg[2]) // 2,
                       (left_avg[3] + right_avg[3]) // 2]
    return left_avg, right_avg, center_line_val, verts, True

def detect_fallback(frame):
    h, w = frame.shape[:2]
    bot_off = int(0.1 * h)
    ROI_TOP, ROI_BOTTOM = int(0.5 * h), h - bot_off
    roi_frame = frame[ROI_TOP:ROI_BOTTOM, :]
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (9, 9), 0), 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    skel = img_as_ubyte(skeletonize(binary // 255))
    contours, _ = cv2.findContours(skel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        return None
    conts = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    try:
        curve1 = cv2.approxPolyDP(conts[0], 5, False).reshape(-1, 2)
        curve2 = cv2.approxPolyDP(conts[1], 5, False).reshape(-1, 2)
    except Exception:
        return None
    num_points = max(len(curve1), len(curve2))
    t = np.linspace(0, 1, num_points)
    curve1_rs = np.array([np.interp(t, np.linspace(0, 1, len(curve1)), curve1[:, i]) for i in range(2)]).T
    curve2_rs = np.array([np.interp(t, np.linspace(0, 1, len(curve2)), curve2[:, i]) for i in range(2)]).T
    midline = (curve1_rs + curve2_rs) / 2.0
    midline[:, 1] += ROI_TOP
    return [int(midline[-1, 0]), int(midline[-1, 1]),
            int(midline[0, 0]), int(midline[0, 1])]

def smooth_line(cur, prev, alpha=smoothing_alpha):
    # Function and algorithm from Bennet Ortiz
    return (alpha * np.array(cur) + (1 - alpha) * np.array(prev)).astype(int).tolist()

def vanishing_point(l_line, r_line):
    if not l_line or not r_line:
        return None
    x1, y1, x2, y2 = l_line
    x3, y3, x4, y4 = r_line
    A1, B1, C1 = y2 - y1, x1 - x2, (y2 - y1) * x1 + (x1 - x2) * y1
    A2, B2, C2 = y4 - y3, x3 - x4, (y4 - y3) * x3 + (x3 - x4) * y3
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-5:
        return None
    return (int((B2 * C1 - B1 * C2) / det), int((A1 * C2 - A2 * C1) / det))

def perspective_transform(frame, l_line, r_line, verts):
    # Function and algorithm from Bennet Ortiz
    global last_vp
    h, w = frame.shape[:2]
    if not l_line or not r_line:
        src = np.float32(verts[0])
    else:
        vp = vanishing_point(l_line, r_line)
        if vp is None:
            vp = last_vp
        margin = 50
        thresh = int(0.5 * h)
        if vp and vp[1] < thresh:
            src = np.float32([[vp[0] - margin, vp[1]],
                              [vp[0] + margin, vp[1]],
                              [r_line[0], h],
                              [l_line[0], h]])
            last_vp = vp
        else:
            src = np.float32(verts[0])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))

def overlay(frame, l_line, r_line, c_line, verts):
    out = frame.copy()
    cv2.polylines(out, [verts], True, (0, 255, 255), 3)
    if l_line:
        cv2.line(out, (l_line[0], l_line[1]), (l_line[2], l_line[3]), (0, 255, 0), 5)
    if r_line:
        cv2.line(out, (r_line[0], r_line[1]), (r_line[2], r_line[3]), (0, 255, 0), 5)
    if c_line:
        cv2.line(out, (c_line[0], c_line[1]), (c_line[2], c_line[3]), (255, 0, 0), 5)
    return out

def is_static(cur, prev, thresh=5):
    if not cur or not prev:
        return False
    return abs(cur[0] - prev[0]) < thresh

def center_line(l_line, r_line, static_left=False):
    if not l_line or not r_line:
        return None
    if static_left:
        return [(2 * l_line[0] + r_line[0]) // 3,
                (2 * l_line[1] + r_line[1]) // 3,
                (2 * l_line[2] + r_line[2]) // 3,
                (2 * l_line[3] + r_line[3]) // 3]
    return [(l_line[0] + r_line[0]) // 2,
            (l_line[1] + r_line[1]) // 2,
            (l_line[2] + r_line[2]) // 2,
            (l_line[3] + r_line[3]) // 2]

def rotate_overlay(overlay_color, alpha, angle):
    """Rotate both the overlay image and its alpha channel by a given angle."""
    (h, w) = overlay_color.shape[:2]
    center_pt = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center_pt, angle, 1.0)
    rot_color = cv2.warpAffine(overlay_color, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT)
    rot_alpha = cv2.warpAffine(alpha, M, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT)
    return rot_color, rot_alpha

import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
import json
import os

# ---------------- Global Variables ----------------
prev_left, prev_right, prev_center = None, None, None
cap = None
arrow_color, arrow_alpha = None, None
# (Assume your lane detection helper functions are defined above this block)

# ----------------==== MAIN APPLICATION (Tkinter UI) ----------------====
def start_main_app():
    global prev_left, prev_right, prev_center, cap, arrow_color, arrow_alpha
    global video_label_top_left, video_label_bottom_left, command_output

    root = tk.Tk()
    root.title("Robot Control")
    root.configure(bg="#e0f7fa")
    root.geometry("800x600")
    root.grid_columnconfigure((0, 1), weight=1)
    root.grid_rowconfigure((0, 1), weight=1)

    # Top Left Quadrant: Raw Webcam Display
    top_left = tk.Frame(root, bg="#000", width=300, height=300)
    top_left.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    video_label_top_left = tk.Label(top_left, bg="#000")
    video_label_top_left.pack(fill="both", expand=True)

    # Bottom Left Quadrant: Processed Webcam Display
    bottom_left = tk.Frame(root, bg="#b0bec5", width=300, height=300)
    bottom_left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
    video_label_bottom_left = tk.Label(bottom_left, bg="#000")
    video_label_bottom_left.pack(fill="both", expand=True)

    # Top Right Quadrant: Controls
    control_frame = tk.Frame(root, bg="#009688", width=300, height=300)
    control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    button_styles = {
        "↑": ("FORWARD", 0, 1, "black", "#1976D2"),
        "←": ("LEFT", 1, 0, "black", "#8E24AA"),
        "STOP": ("STOP", 1, 1, "black", "#D32F2F"),
        "▶": ("PLAY", 1, 2, "black", "#FBC02D"),
        "→": ("RIGHT", 1, 3, "black", "#388E3C"),
        "↓": ("BACKWARD", 2, 1, "black", "#F57C00"),
    }

    def move_robot(direction):
        command_output.insert(tk.END, f"Robot moving: {direction}\n")
        command_output.see(tk.END)

    def stop_robot():
        command_output.insert(tk.END, "Robot STOPPED\n")
        command_output.see(tk.END)

    def start_video():
        update_video()  # Kick off the live webcam stream

    for text, (command, row, col, fg_color, bg_color) in button_styles.items():
        if text == "▶":
            btn = tk.Button(control_frame, text=text, bg=bg_color, fg=fg_color,
                            width=6, height=2, font=("Arial", 14, "bold"),
                            command=start_video)
        else:
            btn = tk.Button(control_frame, text=text, bg=bg_color, fg=fg_color,
                            width=6, height=2, font=("Arial", 14, "bold"),
                            command=lambda c=command: stop_robot() if c == "STOP" else move_robot(c))
        btn.grid(row=row, column=col, padx=5, pady=5)

    # Bottom Right Quadrant: Command Log
    command_output = scrolledtext.ScrolledText(root, height=10, width=50,
                                                 bg="#333", fg="white", font=("Arial", 10))
    command_output.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
    command_output.insert(tk.END, "Command output will appear here...\n")

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Optional: set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Optional: set height

    # Load and prepare the PNG overlay image (arrow)
    arrow_img_path = "/Users/tookd.photo/Desktop/actual.png"  # Update path if needed
    arrow_pil = Image.open(arrow_img_path).convert("RGBA")
    arrow_np = np.array(arrow_pil)
    arrow_np = cv2.cvtColor(arrow_np, cv2.COLOR_RGBA2BGRA)
    scale_factor = 0.35  # Adjust scale as desired
    new_width = int(arrow_np.shape[1] * scale_factor)
    new_height = int(arrow_np.shape[0] * scale_factor)
    arrow_np = cv2.resize(arrow_np, (new_width, new_height))
    if arrow_np.shape[2] == 3:
        arrow_np = cv2.cvtColor(arrow_np, cv2.COLOR_BGR2BGRA)
    b, g, r, a = cv2.split(arrow_np)
    arrow_alpha = a.astype(np.float32) / 255.0  # Normalize alpha to 0-1
    arrow_color = cv2.merge((b, g, r))

    # ------------------ Update Video Function ------------------
    def update_video():
        global prev_left, prev_right, prev_center, cap, arrow_color, arrow_alpha
        ret, frame = cap.read()
        if ret:
            # Top Left: Raw Webcam Display
            frame_top = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_top = Image.fromarray(frame_top).resize((300, 300))
            imgtk_top = ImageTk.PhotoImage(image=img_top)
            video_label_top_left.imgtk = imgtk_top
            video_label_top_left.config(image=imgtk_top)

            # Process frame for lane detection
            frame_proc = undistort(frame)
            h, w = frame_proc.shape[:2]
            bot_off = int(0.1 * h)

            left, right, center, verts, valid = detect_lanes(frame_proc)
            if not valid or center is None:
                fb = detect_fallback(frame_proc)
                if fb:
                    center = fb
                    left = prev_left if prev_left is not None else [int(0.3 * w), h - bot_off, int(0.3 * w), int(0.5 * h)]
                    right = prev_right if prev_right is not None else [int(0.7 * w), h - bot_off, int(0.7 * w), int(0.5 * h)]
                else:
                    pred = predict_center(h)
                    if pred:
                        center = [pred[0][0], pred[0][1], pred[-1][0], pred[-1][1]]
                        for i in range(len(pred) - 1):
                            cv2.line(frame_proc, pred[i], pred[i + 1], (255, 0, 0), 5)
                    else:
                        img_bottom = Image.fromarray(cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)).resize((300, 300))
                        imgtk_bottom = ImageTk.PhotoImage(image=img_bottom)
                        video_label_bottom_left.imgtk = imgtk_bottom
                        video_label_bottom_left.config(image=imgtk_bottom)
                        video_label_top_left.after(30, update_video)
                        return
            else:
                update_hist(center)

            default_left = [int(0.3 * w), h - bot_off, int(0.3 * w), int(0.5 * h)]
            if (not left or left == default_left) and prev_left:
                left = prev_left

            static_left = is_static(left, prev_left) if prev_left and left else False
            new_center = center_line(left, right, static_left)
            if prev_left and left:
                left = smooth_line(left, prev_left)
            if prev_right and right:
                right = smooth_line(right, prev_right)
            center = smooth_line(new_center, prev_center, alpha=0.05) if prev_center and new_center else new_center
            prev_left, prev_right, prev_center = left, right, center

            # Compute turning angle and overlay angle info
            discrete_angle = 0
            if center is not None:
                dx = center[2] - center[0]
                dy = center[3] - center[1]
                computed_angle = np.degrees(np.arctan2(dx, dy))
                if computed_angle > 160:
                    discrete_angle = 0
                elif 120 < computed_angle < 140:
                    discrete_angle = 90
                elif 149 < computed_angle < 158:
                    discrete_angle = -90
                else:
                    discrete_angle = 0
                cv2.putText(frame_proc, f"Angle: {computed_angle:.1f} deg", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Get bird's-eye view (perspective transform) and overlay lanes
            warped = perspective_transform(frame_proc, left, right, verts)
            out_frame = overlay(frame_proc, left, right, center, verts)

            # Rotate and overlay the arrow PNG based on the discrete angle
            rotated_color, rotated_alpha = rotate_overlay(arrow_color, arrow_alpha, discrete_angle)
            pos_x, pos_y = 950, 50  # Adjust position as needed
            oh, ow, _ = rotated_color.shape
            if pos_y + oh <= out_frame.shape[0] and pos_x + ow <= out_frame.shape[1]:
                roi_region = out_frame[pos_y:pos_y + oh, pos_x:pos_x + ow].astype(np.float32)
                oc = rotated_color.astype(np.float32)
                alpha_exp = cv2.merge([rotated_alpha, rotated_alpha, rotated_alpha])
                blended = (1 - alpha_exp) * roi_region + alpha_exp * oc
                out_frame[pos_y:pos_y + oh, pos_x:pos_x + ow] = blended.astype(np.uint8)
            else:
                print("Arrow overlay exceeds frame boundaries. Adjust position or scale.")

            # Bottom Left: Processed Webcam Display
            frame_bottom_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
            img_bottom = Image.fromarray(frame_bottom_rgb).resize((300, 300))
            imgtk_bottom = ImageTk.PhotoImage(image=img_bottom)
            video_label_bottom_left.imgtk = imgtk_bottom
            video_label_bottom_left.config(image=imgtk_bottom)

            # Repeat update after 30ms
            video_label_top_left.after(30, update_video)
        else:
            # If the webcam feed fails, try reinitializing it
            print("Webcam feed not available. Trying to reinitialize...")
            cap.release()
            cap = cv2.VideoCapture(0)
            video_label_top_left.after(30, update_video)

    root.mainloop()


# ----------------==== LOGIN SYSTEM --------------------
# (Assume your accounts loading and saving functions are defined above this block)
login_window = tk.Tk()
login_window.title("Login - Let's Get It Started!")
login_window.geometry("300x250")
login_window.configure(bg="#2C2F33")

tk.Label(login_window, text="Username:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(20, 5))
username_entry = tk.Entry(login_window, bg="#99AAB5", fg="black", font=("Arial", 12))
username_entry.pack(pady=5)

tk.Label(login_window, text="Password:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(10, 5))
password_entry = tk.Entry(login_window, show="*", bg="#99AAB5", fg="black", font=("Arial", 12))
password_entry.pack(pady=5)

login_message = tk.Label(login_window, text="", fg="red", bg="#2C2F33", font=("Arial", 10))
login_message.pack(pady=(10, 5))

def attempt_login():
    username = username_entry.get()
    password = password_entry.get()
    if username in accounts and accounts[username] == password:
        login_message.config(text="Login successful! Lit!", fg="green")
        login_window.destroy()
        start_main_app()
    else:
        login_message.config(text="Invalid creds, try again!", fg="red")

tk.Button(login_window, text="Login", command=attempt_login, bg="#7289DA", fg="white", font=("Arial", 12)).pack(pady=(10, 5))

def open_registration():
    reg_window = tk.Toplevel(login_window)
    reg_window.title("Create Account")
    reg_window.geometry("300x250")
    reg_window.configure(bg="#2C2F33")

    tk.Label(reg_window, text="New Username:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(20, 5))
    reg_username_entry = tk.Entry(reg_window, bg="#99AAB5", fg="black", font=("Arial", 12))
    reg_username_entry.pack(pady=5)

    tk.Label(reg_window, text="New Password:", bg="#2C2F33", fg="white", font=("Arial", 12)).pack(pady=(10, 5))
    reg_password_entry = tk.Entry(reg_window, show="*", bg="#99AAB5", fg="black", font=("Arial", 12))
    reg_password_entry.pack(pady=5)

    reg_message = tk.Label(reg_window, text="", fg="green", bg="#2C2F33", font=("Arial", 10))
    reg_message.pack(pady=(10, 5))

    def register():
        new_username = reg_username_entry.get()
        new_password = reg_password_entry.get()
        if new_username in accounts:
            reg_message.config(text="Username already exists!", fg="red")
        else:
            accounts[new_username] = new_password
            save_accounts(accounts)
            reg_message.config(text="Account created! Go login!", fg="green")
            reg_window.after(1000, reg_window.destroy)

    tk.Button(reg_window, text="Register", command=register, bg="#7289DA", fg="white", font=("Arial", 12)).pack(pady=(10, 5))

tk.Button(login_window, text="Create Account", command=open_registration, bg="#99AAB5", fg="black", font=("Arial", 12)).pack(pady=(10, 5))

login_window.mainloop()
