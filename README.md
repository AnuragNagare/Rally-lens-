# Pickleball AI Analytics

Turn one fixed-camera pickleball video into a broadcast-style analytics video plus
structured stats: shot count, average speed per shot, distance covered, player speed,
time at the kitchen line, a top-down court map and movement heatmaps.

No sensors, no second camera. Pose estimation, ball tracking and a court homography do
the work.

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python run.py --video data/input.mp4 --out output
```

ffmpeg must be on your PATH for H.264 output (without it the video is written as mp4v).
On a 4-core CPU the bundled 20-second clip takes about 7 minutes, almost all of it pose
estimation; a GPU makes that step roughly 20x faster. Pose results are cached in the
output folder, so re-runs that only change rendering take about 90 seconds.

## What you get

| File | Contents |
|---|---|
| `output/analytics.mp4` | 1920x1080 dashboard video |
| `output/stats.json` | Match summary: rallies, shots, speeds, per-player distance and time at the kitchen line |
| `output/shots.csv` | One row per shot: time, player, court zone, hitter and receiver position, distance, average speed |
| `output/tracks.csv` | One row per frame: both players' court position and speed, ball pixel position and state |
| `output/court.json` | The four court corners used for the homography |

## How it works

1. **Court** (`court.py`). A median of 45 frames removes the players and leaves a clean
   background. White lines are found with a Hough transform, the two sidelines and two
   baselines are intersected to get four corners, and those give a homography from
   pixels to metres. The fit is scored by projecting all eight court lines back onto the
   image and checking they land on paint.
2. **Players** (`players.py`). YOLO11 pose runs twice per frame: once on the full frame
   and once zoomed on the far half, because the far player is only about 100 px tall.
   People who never move (officials) and people off the court are dropped. The ankle
   midpoint is projected through the homography, then smoothed.
3. **Ball** (`ball.py`). A classical detector: the ball is a small optic-yellow blob that
   is brighter than the background plate. Candidates are linked into tracklets, and a
   dynamic programme picks the best physically possible chain. Short gaps are
   interpolated and marked as such.
4. **Shots** (`shots.py`). `p(t) = d_far / (d_far + d_near)` is the ball's relative
   position between the two players. Every hit is a turning point of `p`. Bounces barely
   move `p`, so no bounce model is needed. A turning point only counts if the ball is
   within that player's reach.
5. **Speed.** The ball is in the air, where the homography is not valid, so it is never
   projected. A shot's speed is the ground distance from the hitter to the receiver
   divided by the time between the two hits.
6. **Render** (`render.py`). Composes the video panel, 2D court and stats.

## Accuracy on the bundled clip

```bash
python tools/evaluate.py
```

| Measure | Result |
|---|---|
| Shots detected | 16 of 16 hand-labelled hits, no false detections |
| Hit timing error | mean 2.5 frames, worst 6 frames (30 fps) |
| Ball position known | 511 of 606 frames (347 detected, 164 interpolated) |
| Court line fit | 99.7% |

Read these with care: the detector thresholds were tuned on this same clip, and the
labels were made by one person stepping through frames. It shows the pipeline works
here, not how it generalises. Label a few of your own clips and run `evaluate.py`
before quoting a number to a client.

## Known limits

- **Fixed camera only.** The background plate and the homography assume the camera does
  not pan, zoom or cut. Trim replays with `--start` / `--end`.
- **Singles only.** One player is tracked per side. Doubles needs a proper multi-object
  tracker and a new rule for who hit the ball.
- **Shot speed is an average**, including the slowdown after a bounce. It reads lower
  than a radar gun and is labelled "avg" on screen.
- **Ball colour.** The classical detector expects an optic-yellow ball. For other
  colours, lighting changes or a moving camera, train a YOLO ball model (below).
- **Feet out of frame.** When the near player's feet leave the bottom of the image, the
  position falls back to the bottom of their box and is less accurate.
- **Ball feeds between points** are dropped only when the opponent is out of shot.

## Using your own footage

Best results: camera behind a baseline, elevated, whole court visible, 1080p, 30 fps or
more, tripod-mounted.

```bash
python run.py --video my_match.mp4 --out out_my_match --near-name "Asha" --far-name "Ravi"
```

If the log reports a poor court line fit, calibrate by hand and pass the result:

```bash
python tools/pick_court_points.py --video my_match.mp4 --out court.json
```

```bash
python run.py --video my_match.mp4 --court-json court.json
```

## Training a custom ball model

1. Export about 1,500 to 3,000 frames across different courts and lighting.
2. Label the ball as a single class `ball` in Roboflow or CVAT. `output/tracks.csv` from
   the classical detector gives you pre-labels to correct instead of starting from zero.
3. Fine-tune at high resolution, since the ball is only a few pixels wide:

```bash
yolo detect train model=yolo11s.pt data=ball.yaml imgsz=1280 epochs=100
```

4. Use it in place of the classical detector:

```bash
python run.py --video my_match.mp4 --ball-weights runs/detect/train/weights/best.pt
```


## Layout

```
run.py                       command-line entry point
pickle_analytics/
  court.py                   court model, line detection, homography
  players.py                 pose inference, player selection, smoothing
  ball.py                    ball candidates, tracklets, chain selection
  shots.py                   rallies, hits, per-shot speed
  render.py                  dashboard video
  pipeline.py                orchestration and stats files
tools/
  evaluate.py                score shots against hand labels
  pick_court_points.py       manual court calibration
data/                        sample clip and its hit labels
output/                      results from the sample clip
```





https://github.com/user-attachments/assets/fa4e4788-9108-492b-a08a-ae9dad904d01


