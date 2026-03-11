import cv2
import json
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def investigate(video_path, json_path, chamber_num):
    print(f"Loading data for {video_path} - Chamber {chamber_num}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    c_name = f"Chamber_{chamber_num}"
    if c_name not in data:
        print(f"Error: {c_name} not found in {json_path}")
        return
        
    track = data[c_name]
    
    # Extract data for plotting
    frames, speeds = [], []
    speed_data_tuples = [] # To sort and find top 10
    
    for f_idx, state in track.items():
        if f_idx == "Summary": continue
        
        f_int = int(f_idx)
        speed = state['speed_mm']
        frames.append(f_int)
        speeds.append(speed)
        speed_data_tuples.append((f_int, speed))
        
    # --- PART 1: DIAGNOSTICS & PLOTTING ---
    # Sort descending to find the top 10 fastest speeds
    speed_data_tuples.sort(key=lambda x: x[1], reverse=True)
    print("\n--- TOP 10 HIGHEST SPEEDS ---")
    for i in range(min(10, len(speed_data_tuples))):
        f_idx, spd = speed_data_tuples[i]
        print(f"  {i+1}. Frame {f_idx}: {spd} mm/s")
    
    print("\n[!] A scatterplot has been generated. Please close the plot window to continue to the video investigator.")
    
    plt.figure(figsize=(10, 5))
    plt.scatter(frames, speeds, s=2, alpha=0.5, color='blue')
    plt.title(f"Speed vs Frame Index - {c_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Estimated Speed (mm/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show() # This blocks execution until the user closes the window

    # --- PART 2: INTERACTIVE VIDEO EXTRACTION ---
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate chamber bounding boxes ONCE to save time
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    b, g, r = frame[:, :, 0]/255.0, frame[:, :, 1]/255.0, frame[:, :, 2]/255.0
    mask = ((r - 0.5) + (g - 0.5) + (b - 0.5)) > 0 
    clean_mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
    
    chambers = []
    max_dim = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        max_dim = max(max_dim, w, h)
        M = cv2.moments(c)
        if M['m00'] != 0:
            chambers.append({'cx': int(M['m10'] / M['m00']), 'cy': int(M['m01'] / M['m00'])})
            
    square_size = max_dim + 20
    half_size = square_size // 2
    
    chambers = sorted(chambers, key=lambda d: d['cy'])
    sorted_chambers = sorted(chambers[:3], key=lambda d: d['cx']) + sorted(chambers[3:], key=lambda d: d['cx'])
    target_chamber = sorted_chambers[chamber_num - 1]
    cx, cy = target_chamber['cx'], target_chamber['cy']
    
    frame_offset = int(fps * 2.5) # 2.5 seconds in each direction = 5 second video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    print("\n--- INTERACTIVE INVESTIGATOR ---")
    while True:
        user_input = input("Enter a frame index to investigate (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            break
            
        try:
            target_frame = int(user_input)
        except ValueError:
            print("Please enter a valid number or 'q'.")
            continue
            
        start_frame = max(0, target_frame - frame_offset)
        end_frame = target_frame + frame_offset
        
        out_name = f"INVESTIGATION_Chamber_{chamber_num}_Frame_{target_frame}.mp4"
        writer = cv2.VideoWriter(out_name, fourcc, fps/2, (square_size, square_size)) # Half speed
        
        print(f"  -> Rendering {out_name} (Frames {start_frame} to {end_frame})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        recent_pts = []
        for current_frame in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break
            
            y1, y2 = cy - half_size, cy + half_size
            x1, x2 = cx - half_size, cx + half_size
            crop = frame[y1:y2, x1:x2].copy()
            
            frame_str = str(current_frame)
            if frame_str in track:
                pt = track[frame_str]
                x_px, y_px = pt['x_px'], pt['y_px']
                recent_pts.append((x_px, y_px))
                if len(recent_pts) > 30:
                    recent_pts.pop(0)
                    
                for i in range(1, len(recent_pts)):
                    cv2.line(crop, recent_pts[i-1], recent_pts[i], (0, 255, 255), 1)
                    
                # Highlight the EXACT target frame with a green circle instead of red
                if current_frame == target_frame:
                    cv2.circle(crop, (x_px, y_px), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(crop, (x_px, y_px), 3, (0, 0, 255), -1)
                
                cv2.putText(crop, f"Frame: {current_frame}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(crop, f"Speed: {pt['speed_mm']} mm/s", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            writer.write(crop)
            
        writer.release()
        print(f"  -> Done!")

    cap.release()
    print("Exiting investigator.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video", required=True)
    parser.add_argument("-j", "--json", required=True)
    parser.add_argument("-c", "--chamber", type=int, required=True)
    args = parser.parse_args()
    investigate(args.video, args.json, args.chamber)