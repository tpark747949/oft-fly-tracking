import cv2
import numpy as np
import argparse
import os
import json
import glob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as colors

def process_video(input_path, out_dir, active_chambers, args, global_data):
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"  [!] Error: Could not open {filename}")
        return

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"  [!] CRITICAL ERROR: File '{filename}' is corrupted or missing its 'moov' atom. Skipping.")
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = 1.0 / fps if fps > 0 else 0.033
    
    print(f"  -> Calculating Median Background...")
    frame_indices = np.random.choice(total_frames, min(50, total_frames), replace=False)
    bg_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret: bg_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    median_bg = np.median(bg_frames, axis=0).astype(np.uint8)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    b, g, r = first_frame[:, :, 0]/255.0, first_frame[:, :, 1]/255.0, first_frame[:, :, 2]/255.0
    mask = ((r - args.r) + (g - args.g) + (b - args.b)) > 0
    clean_mask = cv2.morphologyEx((mask * 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
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
            _, radius = cv2.minEnclosingCircle(c)
            chambers_info.append({
                'cx': int(M['m10'] / M['m00']), 'cy': int(M['m01'] / M['m00']), 
                'px_per_mm': (radius * 2) / 10.0
            })

    square_size = max_dim + 20
    half_size = square_size // 2

    chambers_info = sorted(chambers_info, key=lambda d: d['cy'])
    sorted_chambers = sorted(chambers_info[:3], key=lambda d: d['cx']) + sorted(chambers_info[3:], key=lambda d: d['cx'])

    writers, heatmaps, tracking_data, last_state, background_crops = {}, {}, {}, {}, {}
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for i, chamber in enumerate(sorted_chambers):
        c_num = i + 1
        if c_num not in active_chambers: continue
        
        c_name = f"Chamber_{c_num}"
        tracking_data[c_name] = {"Summary": {"total_distance_mm": 0.0}}
        last_state[c_name] = None
        
        cx, cy = chamber['cx'], chamber['cy']
        y1, y2, x1, x2 = cy - half_size, cy + half_size, cx - half_size, cx + half_size
        background_crops[c_name] = median_bg[y1:y2, x1:x2].copy()

        if args.labelled:
            writers[c_name] = cv2.VideoWriter(os.path.join(out_dir, f"{name}_{c_name}_LABELLED.mp4"), fourcc, fps, (square_size, square_size))
        if args.heatmap:
            heatmaps[c_name] = np.zeros((square_size, square_size), dtype=np.float32)

    print(f"  -> Tracking active chambers...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0
    closing_kernel = np.ones((7, 7), np.uint8) 
    jitter_px = 0.0

    while True:
        ret, frame = cap.read()
        if not ret: break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(median_bg, current_gray)
        _, thresh_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, closing_kernel)

        if args.labelled:
            b, g, r = frame[:, :, 0]/255.0, frame[:, :, 1]/255.0, frame[:, :, 2]/255.0
            bw_frame = np.zeros_like(frame)
            bw_frame[((r - args.r) + (g - args.g) + (b - args.b)) > 0] = [255, 255, 255]

        for i, chamber in enumerate(sorted_chambers):
            c_num = i + 1
            if c_num not in active_chambers: continue
            
            c_name = f"Chamber_{c_num}"
            cx, cy, px_per_mm = chamber['cx'], chamber['cy'], chamber['px_per_mm']
            y1, y2, x1, x2 = cy - half_size, cy + half_size, cx - half_size, cx + half_size
            
            crop_diff = thresh_diff[y1:y2, x1:x2]
            fly_contours, _ = cv2.findContours(crop_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            x_px, y_px, x_mm, y_mm, dist_mm = 0, 0, 0, 0, 0
            vx_mm, vy_mm, speed_mm = 0, 0, 0
            found_fly = False
            
            if fly_contours:
                fly = max(fly_contours, key=cv2.contourArea)
                if cv2.contourArea(fly) > 5:
                    M = cv2.moments(fly)
                    if M['m00'] != 0:
                        raw_x, raw_y = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                        found_fly = True
                        
                        last = last_state[c_name]
                        if last is not None:
                            dist_px = np.sqrt((raw_x - last['x_px'])**2 + (raw_y - last['y_px'])**2)
                            if dist_px < jitter_px:
                                raw_x, raw_y = last['x_px'], last['y_px']
                                
                            vx_px = (raw_x - last['x_px']) / dt
                            vy_px = (raw_y - last['y_px']) / dt 
                            vx_mm = vx_px / px_per_mm
                            vy_mm = -vy_px / px_per_mm 
                            
                            speed_mm = np.sqrt(vx_mm**2 + vy_mm**2) # Teleportation filter removed
                            global_data['all_speeds'].append(speed_mm)
                        
                        x_px, y_px = raw_x, raw_y
                        x_mm = (x_px - half_size) / px_per_mm
                        y_mm = (half_size - y_px) / px_per_mm 
                        dist_mm = np.sqrt(x_mm**2 + y_mm**2)

                        state = {
                            "time_s": round(frame_count * dt, 3),
                            "x_px": x_px, "y_px": y_px, "x_mm": round(x_mm, 3), "y_mm": round(y_mm, 3),
                            "dist_mm": round(dist_mm, 3), "vx_mm": round(vx_mm, 3), "vy_mm": round(vy_mm, 3),
                            "speed_mm": round(speed_mm, 3)
                        }
                        tracking_data[c_name][frame_count] = state
                        last_state[c_name] = state
                        tracking_data[c_name]["Summary"]["total_distance_mm"] += (speed_mm * dt)

                        if args.heatmap: heatmaps[c_name][y_px, x_px] += 1

            if args.labelled:
                crop_bw = bw_frame[y1:y2, x1:x2].copy()
                cv2.drawMarker(crop_bw, (half_size, half_size), (200, 200, 200), cv2.MARKER_CROSS, 10, 1)
                if found_fly: cv2.circle(crop_bw, (x_px, y_px), 3, (0, 0, 255), -1)
                writers[c_name].write(crop_bw)

        frame_count += 1

    cap.release()
    if args.labelled:
        for w in writers.values(): w.release()

    if args.heatmap:
        for c_name, hm in heatmaps.items():
            hm_blurred = cv2.GaussianBlur(hm, (15, 15), 0)
            hm_norm = np.uint8(255 * (hm_blurred / np.max(hm_blurred))) if np.max(hm_blurred) > 0 else np.uint8(hm_blurred)
            cv2.imwrite(os.path.join(out_dir, f"{name}_{c_name}_HEATMAP.jpg"), cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET))

    tracking_data["Summary"] = {
        c: round(tracking_data[c]["Summary"]["total_distance_mm"], 2) for c in tracking_data if c != "Summary"
    }

    with open(os.path.join(out_dir, f"{name}_tracking.json"), 'w') as f:
        json.dump(tracking_data, f, indent=4)

    global_data['videos'][name] = {'tracking': tracking_data, 'backgrounds': background_crops, 'size': square_size}
    print(f"  -> Finished {filename}")

def main():
    parser = argparse.ArgumentParser(description="Fly Tracking Suite Pro")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default=".")
    parser.add_argument("-r", type=float, default=0.5)
    parser.add_argument("-g", type=float, default=0.5)
    parser.add_argument("-b", type=float, default=0.5)
    parser.add_argument("-l", "--labelled", action="store_true")
    parser.add_argument("-t", "--trace", action="store_true")
    parser.add_argument("-hm", "--heatmap", action="store_true")
    parser.add_argument("-vmax", type=float, default=40.0, help="Manual speed limit for red-out in traces (default: 40)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    videos = glob.glob(os.path.join(args.input, "*.mp4")) if os.path.isdir(args.input) else [args.input]

    print("\n=== CONFIGURATION SETUP ===")
    active_chambers_dict = {}
    for vid in videos:
        basename = os.path.basename(vid)
        ans = input(f"Exclude chambers for {basename}? (Enter comma-separated numbers 1-6, or press Enter to keep all): ")
        excluded = [int(x.strip()) for x in ans.split(',') if x.strip().isdigit()]
        active_chambers_dict[vid] = [c for c in range(1, 7) if c not in excluded]

    global_data = {'all_speeds': [], 'videos': {}}
    print(f"\n=== PHASE 1: Processing {len(videos)} videos ===")
    for vid in videos:
        process_video(vid, args.output, active_chambers_dict[vid], args, global_data)

    if args.trace:
        print("\n=== PHASE 2: Generating Continuous, Transparent Traces ===")
        max_speed = args.vmax
        print(f"Red saturation clamped at {max_speed:.2f} mm/s")

        for name, data in global_data['videos'].items():
            for c_name, track in data['tracking'].items():
                if c_name == "Summary": continue
                
                size = data['size']
                bg = data['backgrounds'][c_name]
                
                # Setup Matplotlib for transparent, continuous rendering
                fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
                ax.imshow(bg, cmap='gray', extent=[0, size, size, 0])
                
                points, speeds = [], []
                frames = sorted([f for f in track.keys() if f != "Summary"], key=int)
                
                for i in frames:
                    pt = track[i]
                    points.append([pt['x_px'], pt['y_px']])
                    speeds.append(pt['speed_mm'])

                # Create line segments
                points = np.array(points)
                segments = np.concatenate([points[:-1, None, :], points[1:, None, :]], axis=1)
                
                # Create a continuous colormap and apply transparency
                norm = colors.LogNorm(vmin=max(0.1, min(speeds)), vmax=max_speed)
                # norm = colors.Normalize(vmin=0, vmax=max_speed)
                lc = LineCollection(segments, cmap='rainbow', norm=norm, alpha=0.5, linewidths=1.0)
                lc.set_array(np.array(speeds))
                ax.add_collection(lc)
                
                ax.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.margins(0,0)
                
                out_file = os.path.join(args.output, f"{name}_{c_name}_TRACE.jpg")
                plt.savefig(out_file, pad_inches=0, bbox_inches='tight')
                plt.close(fig)

    print("\nAll processing complete!")

if __name__ == "__main__":
    main()