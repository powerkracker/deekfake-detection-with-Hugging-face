import os
import cv2
import torch
from transformers import AutoImageProcessor, ViTForImageClassification
import torch.nn.functional as F
from PIL import Image
import time
import json

# === Load model & processor ===
MODEL_NAME = "shivani1511/deepfake-image-detector-new-latest-v2"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# === Video setup ===
input_video = '01__talking_against_wall.mp4'
basename = os.path.splitext(os.path.basename(input_video))[0]
output_video = f"{basename}_output_with_marks.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# === Config ===
threshold = 0.8
frame_interval = 1
adjacent_frames = 2
mark_window = 0

# === Counters ===
frame_count = 0
analyzed_frames = 0
real_count = 0
fake_count = 0

# Map class indices manually (LABEL_0=Fake, LABEL_1=Real)
fake_index = 0
real_index = 1

# Start total timer
start_time = time.time()
print(f"Processing video: {total_frames} frames, analyzing every {frame_interval} frame(s)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    analyze_this = (frame_count % frame_interval == 0)
    is_fake = False

    frame_start = time.time()

    if analyze_this:
        analyzed_frames += 1

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=pil_img, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()

        is_fake = probs[fake_index] > threshold

        if is_fake:
            mark_window = adjacent_frames

        # Count for stats
        if is_fake:
            fake_count += 1
        else:
            real_count += 1
    else:
        if mark_window > 0:
            is_fake = True
            mark_window -= 1

    # Draw marks
    if is_fake:
        cv2.putText(frame, "FAKE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 20)

    # Write frame
    out.write(frame)

    frame_end = time.time()
    if analyze_this:
        print(f"Frame {frame_count}/{total_frames} (analyzed): {1000*(frame_end - frame_start):.1f} ms")
    else:
        print(f"Frame {frame_count}/{total_frames} (skipped)")

cap.release()
out.release()

# End total timer
end_time = time.time()
total_time = end_time - start_time

# === Save JSON stats ===
stats_file = f"{basename}_deepfake_stats.json"
stats = {
    "total_frames": total_frames,
    "analyzed_frames": analyzed_frames,
    "real_frames": real_count,
    "fake_frames": fake_count,
    "real_percent": round(100 * real_count / analyzed_frames, 2) if analyzed_frames > 0 else 0.0,
    "fake_percent": round(100 * fake_count / analyzed_frames, 2) if analyzed_frames > 0 else 0.0,
    "total_time_sec": round(total_time, 2),
    "time_per_analyzed_frame_sec": round(total_time / analyzed_frames, 4) if analyzed_frames > 0 else None,
    "time_per_total_frame_sec": round(total_time / total_frames, 4) if total_frames > 0 else None
}

with open(stats_file, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\n✅ Processed {frame_count} frames")
print(f"📊 Stats saved as '{stats_file}'")
