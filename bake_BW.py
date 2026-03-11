import cv2
import numpy as np
import argparse
import os
import json

def process_video(input_path, r_mid, g_mid, b_mid, diagnostic, trace):
    filename, ext = os.path.splitext(input_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- STEP 1: CALCULATE MEDIAN BACKGROUND ---
    print("Calculating Median Background...")
    frame_indices = np.random.choice(total_frames, min(50, total_frames), replace=False)
    bg_frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            
    median_bg = np.median(bg_frames, axis=0).astype(np.uint8)

    # --- STEP 2: FIND CHAMBERS & CALCULATE SCALE ---
    print("Locating chambers and calculating physical scale...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    
    b, g, r = first_frame[:, :, 0]/255.0, first_frame[:, :, 1]/255.0, first_frame[:, :, 2]/255.0
    mask = ((r - r_mid) + (g - g_mid) + (b - b_mid)) > 0
    mask_uint8 = (mask * 255).astype(np.uint8)

    clean_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:6]
    
    chambers_info = []
    max_dim = 0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        max_dim = max(max_dim, w, h)
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Calculate physical scale: diameter is exactly 10 mm
            _, radius = cv2.minEnclosingCircle(c)
            px_per_mm = (radius * 2) / 10.0
            
            chambers_info.append({'cx': cx, 'cy': cy, 'px_per_mm': px_per_mm})

    square_size = max_dim + 20
    half_size = square_size // 2

    # Sort chambers spatially
    chambers_info = sorted(chambers_info, key=lambda d: d['cy'])
    top_row = sorted(chambers_info[:3], key=lambda d: d['cx'])
    bottom_row = sorted(chambers_info[3:], key=lambda d: d['cx'])
    sorted_chambers = top_row + bottom_row

    # --- STEP 3: INITIALIZE DATA ---
    writers = []
    tracking_data = {}
    last_positions = {} 
    
    if diagnostic:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Diagnostic mode ON: Generating tracking videos...")

    for i in range(len(sorted_chambers)):
        chamber_name = f"Chamber_{i+1}"
        tracking_data[chamber_name] = {}
        last_positions[chamber_name] = None
        if diagnostic:
            writers.append(cv2.VideoWriter(f"{filename}_{chamber_name}_DIAGNOSTIC{ext}", fourcc, fps, (square_size, square_size)))

    print("Tracking flies...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    closing_kernel = np.ones((7, 7), np.uint8) 
    jitter_threshold = 2.5 # Pixels

    # --- STEP 4: PROCESS FRAME BY FRAME ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(median_bg, current_gray)
        _, thresh_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, closing_kernel)

        if diagnostic:
            b, g, r = frame[:, :, 0]/255.0, frame[:, :, 1]/255.0, frame[:, :, 2]/255.0
            bw_mask = ((r - r_mid) + (g - g_mid) + (b - b_mid)) > 0
            bw_frame = np.zeros_like(frame)
            bw_frame[bw_mask] = [255, 255, 255]

        for i, chamber in enumerate(sorted_chambers):
            cx, cy = chamber['cx'], chamber['cy']
            px_per_mm = chamber['px_per_mm']
            chamber_name = f"Chamber_{i+1}"
            
            y1, y2 = cy - half_size, cy + half_size
            x1, x2 = cx - half_size, cx + half_size
            
            crop_diff = thresh_diff[y1:y2, x1:x2]
            fly_contours, _ = cv2.findContours(crop_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if fly_contours:
                fly = max(fly_contours, key=cv2.contourArea)
                if cv2.contourArea(fly) > 5:
                    M = cv2.moments(fly)
                    if M['m00'] != 0:
                        fly_x = int(M['m10'] / M['m00'])
                        fly_y = int(M['m01'] / M['m00'])
                        
                        # Jitter Reduction
                        last_pos = last_positions[chamber_name]
                        if last_pos is not None:
                            dist_px = np.sqrt((fly_x - last_pos[0])**2 + (fly_y - last_pos[1])**2)
                            if dist_px < jitter_threshold:
                                fly_x, fly_y = last_pos[0], last_pos[1]
                        
                        last_positions[chamber_name] = (fly_x, fly_y)
                        
                        # --- PHYSICAL MATH LOGIC (mm) ---
                        # Center of the crop is exactly at (half_size, half_size)
                        # We invert the Y logic so that "Up" is mathematically positive
                        x_mm = (fly_x - half_size) / px_per_mm
                        y_mm = (half_size - fly_y) / px_per_mm 
                        dist_mm = np.sqrt(x_mm**2 + y_mm**2)
                        
                        tracking_data[chamber_name][frame_count] = {
                            "x_px": int(fly_x), 
                            "y_px": int(fly_y),
                            "x_mm": round(x_mm, 4),
                            "y_mm": round(y_mm, 4),
                            "dist_mm": round(dist_mm, 4)
                        }

            if diagnostic:
                crop_bw = bw_frame[y1:y2, x1:x2].copy()
                if crop_bw.shape[0] != square_size or crop_bw.shape[1] != square_size:
                     padded_crop = np.zeros((square_size, square_size, 3), dtype=np.uint8)
                     padded_crop[0:crop_bw.shape[0], 0:crop_bw.shape[1]] = crop_bw
                     crop_bw = padded_crop
                
                if last_positions[chamber_name] is not None:
                    fx, fy = last_positions[chamber_name]
                    cv2.circle(crop_bw, (fx, fy), 3, (0, 0, 255), -1)
                    
                writers[i].write(crop_bw)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames...")

    cap.release()
    if diagnostic:
        for w in writers:
            w.release()

    # --- STEP 5: DRAW TRACE IMAGES ---
    if trace:
        print("Generating traced path images...")
        for i, chamber in enumerate(sorted_chambers):
            cx, cy = chamber['cx'], chamber['cy']
            chamber_name = f"Chamber_{i+1}"
            
            # Crop the median background to draw on
            y1, y2 = cy - half_size, cy + half_size
            x1, x2 = cx - half_size, cx + half_size
            trace_bg = median_bg[y1:y2, x1:x2].copy()
            
            if trace_bg.shape[0] != square_size or trace_bg.shape[1] != square_size:
                 padded_bg = np.zeros((square_size, square_size), dtype=np.uint8)
                 padded_bg[0:trace_bg.shape[0], 0:trace_bg.shape[1]] = trace_bg
                 trace_bg = padded_bg
            
            trace_color = cv2.cvtColor(trace_bg, cv2.COLOR_GRAY2BGR)
            chamber_track = tracking_data[chamber_name]
            
            # Sort frames sequentially to draw continuous lines
            frames = sorted([f for f in chamber_track.keys()])
            for j in range(1, len(frames)):
                pt1 = (chamber_track[frames[j-1]]['x_px'], chamber_track[frames[j-1]]['y_px'])
                pt2 = (chamber_track[frames[j]]['x_px'], chamber_track[frames[j]]['y_px'])
                cv2.line(trace_color, pt1, pt2, (255, 0, 0), 1) # Blue line
            
            cv2.imwrite(f"{filename}_{chamber_name}_trace.jpg", trace_color)
        
    # Export JSON
    json_path = f"{filename}_tracking.json"
    with open(json_path, 'w') as f:
        json.dump(tracking_data, f, indent=4)
        
    print(f"Success! Data saved to '{json_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to input video")
    parser.add_argument("-r", type=float, default=0.5, help="Red threshold")
    parser.add_argument("-g", type=float, default=0.5, help="Green threshold")
    parser.add_argument("-b", type=float, default=0.5, help="Blue threshold")
    parser.add_argument("-d", "--diagnostic", action="store_true", help="Output diagnostic videos")
    
    # NEW TRACE FLAG
    parser.add_argument("-t", "--trace", action="store_true", help="Output images tracing the fly's path")
    
    args = parser.parse_args()
    process_video(args.input, args.r, args.g, args.b, args.diagnostic, args.trace)