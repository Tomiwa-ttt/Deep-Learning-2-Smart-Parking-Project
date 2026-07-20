#!/bin/bash
set -e
cd "$(dirname "$0")"

FRAMES=video_frames
WORK=video_work
mkdir -p $WORK

# concat list: image, duration (seconds)
cat > $WORK/concat_list.txt << 'EOF'
file '../video_frames/00_title.png'
duration 3.0
file '../video_frames/01a_raw.png'
duration 2.2
file '../video_frames/01b_annotated.png'
duration 3.5
file '../video_frames/02a_raw.png'
duration 2.2
file '../video_frames/02b_annotated.png'
duration 3.5
file '../video_frames/03a_raw.png'
duration 2.2
file '../video_frames/03b_annotated.png'
duration 3.5
file '../video_frames/90_stats.png'
duration 3.5
file '../video_frames/91_transition.png'
duration 2.5
file '../video_frames/91_transition.png'
duration 0.04
EOF

# Build the photo-slideshow section (silent, 480x900, h264)
ffmpeg -y -f concat -safe 0 -i $WORK/concat_list.txt -vsync vfr \
  -vf "fps=25,format=yuv420p" -c:v libx264 -movflags +faststart \
  $WORK/section_photos.mp4

# Re-encode the app clip to guarantee identical codec params before concat
ffmpeg -y -i campuspark_demo.mp4 -vf "fps=25,scale=480:900,format=yuv420p" \
  -c:v libx264 -movflags +faststart $WORK/section_app.mp4

# Concatenate photo section + app section (stream copy, same codec now)
cat > $WORK/final_concat.txt << EOF
file 'section_photos.mp4'
file 'section_app.mp4'
EOF

ffmpeg -y -f concat -safe 0 -i $WORK/final_concat.txt -c copy campuspark_full_demo.mp4

echo "DONE"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 campuspark_full_demo.mp4
