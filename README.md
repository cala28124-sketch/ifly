# Fly Brain Collision Warning

A fruit fly's visual system and brain, simulated from real connectome data, watch the road behind a cyclist
and play a fly's buzz in the rider's headset when a vehicle is on a collision course. It uses the same
looming-escape circuit a real fly uses to dodge a swatter, and every warning names the neurons that fired
(e.g. `LC4 > LPLC2 > DNp01`). Built at ShellHacks 2026 and tested on the Waymo Open Dataset.

## Pipeline

```
Raspberry Pi (camera + headset)                 Laptop (all the modeling)
pi/sender.py                                     laptop/receiver.py
  webcam -> 320x240 color JPEG --- port 5555 -->   gray 160x120, 1.25x zoom
                                                   laptop/eyes.py   flyvis optic lobe (~45,000 neurons)
                                                                    T4/T5 motion -> LPLC2-like looming units,
                                                                    looming cut 70% while the camera turns
                                                   laptop/brain.py  flybrain MaleCNS brain (166,700 neurons):
                                                                    loom -> LPLC2 + LC4 -> giant fiber (DNp01)
                                                   vehicle/detector.py  YOLO vehicle boxes (display only)
pi/commands.py  <--- {"escape", "turn"} port 5556 ---
  fly buzz in the left / right / both ears       laptop/viewer.py  camera view, circuit panel, brain map
```

About 6 decisions per second; ~150 ms per decision on a laptop CPU. The fly escapes when its giant fiber
spikes twice in a chunk. The fly's neurons and wiring are used unchanged; our code prepares the input, reads
out T4/T5 and LPLC2-style looming, and sets the thresholds in `config.py`.

## Repo layout

| Path | What it is |
| --- | --- |
| `config.py` | Every setting: laptop IP, ports, sizes, threshold, fake/real switches, headset |
| `laptop/receiver.py` | Main loop on the laptop (frames in, warnings out) |
| `laptop/eyes.py`, `laptop/brain.py` | Fly eyes (flyvis) and fly brain (flybrain); each has a fake mode |
| `laptop/viewer.py`, `laptop/brain_map.py` | Camera view + circuit panel window, whole-brain map window |
| `pi/sender.py`, `pi/commands.py` | Pi camera sender; headset listener (the fly buzz) |
| `pi/__main__.py` | `python -m pi` runs both Pi programs together |
| `vehicle/detector.py` | YOLO11n vehicle detector (weights in `vehicle/yolo11n.pt`) |
| `laptop/waymo_player.py`, `laptop/waymo_sender.py`, `laptop/score_waymo.py` | Waymo playback, feeding clips to the receiver, scoring |
| `tools/` | Direction tuning of the eye model, synthetic looming test, flyvis check |

## Setup

### Laptop (Mac, Linux or Windows)

Needs Python 3.12 (flyvis doesn't support 3.13+) and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cala28124-sketch/ifly.git
cd ifly
uv venv --python 3.12
source .venv/bin/activate            # Windows: .venv\Scripts\activate
uv pip install opencv-python imagezmq pyzmq numpy flyvis flybrain ultralytics pandas pyarrow
flyvis download-pretrained           # fly eye models, one time
flybrain download                    # fly brain data (~260 MB, one time)
```

Check it: `python -c "from laptop.eyes import FlyEyes; from laptop.brain import FlyBrainModel; FlyEyes(fake=False); FlyBrainModel(fake=False); print('OK')"`

Without flyvis/flybrain, set `FAKE_EYES = True` and `FAKE_BRAIN = True` in `config.py`; without ultralytics, set `VEHICLE_CHECK = False`.

### Raspberry Pi 4 (Raspberry Pi OS)

```bash
sudo apt install -y python3-opencv v4l-utils git alsa-utils
python3 -m venv --system-site-packages ~/fly-venv
source ~/fly-venv/bin/activate
pip install imagezmq pyzmq
git clone https://github.com/cala28124-sketch/ifly.git ~/ifly-git
```

Headset: find its name with `aplay -l` (e.g. `card 4: G [...]`) and set `AUDIO_DEVICE = "plughw:CARD=G,DEV=0"` in
`config.py`. Test with `python -m pi.commands --test` (buzzes left, right, then both).

## Run

Run everything from the repo root with the virtual environment active. Stop with Esc in the laptop window and
Ctrl+C in terminals.

### Laptop only (its own webcam)

Set `LAPTOP_IP = "127.0.0.1"` in `config.py` (don't commit it), then in three terminals:

```bash
python -m laptop.receiver      # wait ~15 s for eyes + brain + YOLO to load
python -m pi.sender            # sends the laptop's camera 0 (change the index in pi/sender.py if needed)
python -m pi.commands          # optional: the fly buzz on the laptop's speakers
```

### Pi + laptop

1. Put both on the same network and find the laptop's IP (`ipconfig getifaddr en0` on a Mac, `ipconfig` on Windows).
2. On the Pi, set `LAPTOP_IP` in `config.py` to that IP.
3. Laptop: `python -m laptop.receiver` (wait ~15 s).
4. Pi: `cd ~/ifly-git && source ~/fly-venv/bin/activate && python -m pi`

Push something straight toward the camera: WARN appears and the headset buzzes in the ear on that side. Sideways
motion and turns stay quiet ("TURNING" shows while looming is reduced).

## Testing on Waymo footage

Needs a Google account that accepted the [Waymo Open Dataset](https://waymo.com/open/) license, the Google
Cloud CLI (`gcloud auth login`, and `gcloud components install gcloud-crc32c` so downloads pass their checksum).

```bash
mkdir -p waymo/camera_image waymo/lidar_box
gcloud storage cp -n "gs://waymo_open_dataset_v_2_0_1/validation/lidar_box/*.parquet" waymo/lidar_box/
gcloud storage cp -n gs://waymo_open_dataset_v_2_0_1/validation/camera_image/SEGMENT.parquet waymo/camera_image/

python -m laptop.receiver --fps 10 --chunk 2 --log waymo/results/SEGMENT.csv    # terminal 1
python -m laptop.waymo_sender SEGMENT                                            # terminal 2 (--realtime to watch)
python -m laptop.score_waymo                                                     # hits, lead time, false alarms
```

The score uses Waymo's 3D labels: a collision course is a vehicle clearly in the car's path that would reach
it within 3 s (sharp turns excluded); close passes are reported separately. `waymo/` is git-ignored.

## Key settings (`config.py`)

| Setting | Default | Meaning |
| --- | --- | --- |
| `WARN_THRESHOLD` | 0.4 | Loom level where the fly escapes (lower = earlier, more false alarms) |
| `EYES_ZOOM` | 1.25 | Zoom into the frame centre (earlier warnings, narrower view) |
| `CAMERA_FPS`, `CHUNK` | 30, 5 | Live frame rate and frames per decision |
| `FAKE_EYES`, `FAKE_BRAIN` | False | Use the simple stand-ins instead of the fly models |
| `VEHICLE_CHECK` | True | YOLO boxes in the viewer |
| `AUDIO_DEVICE`, `SWAP_EARS` | headset, True | Pi audio output; rear camera flips left and right |

## Troubleshooting

- **No window on the laptop:** the receiver waits for frames; check the sender's terminal and `LAPTOP_IP`.
- **`Address already in use`:** an old receiver is still running (`pkill -f laptop.receiver`).
- **`camera 0 not available`:** allow camera access for the terminal app, or try camera index 1.
- **No sound on the Pi (`audio open error: 524`):** audio is going to HDMI; set `AUDIO_DEVICE` from `aplay -l`.
- **Camera shake or flicker warnings:** lock auto-exposure with `v4l2-ctl -d /dev/video0 --list-ctrls` and the exposure controls it lists.

## Known limitations

The fly warns late (cars under ~13 m; its eye has 721 facets), ignores cars that will pass beside it, and
turns looming down during turns. On 8 Waymo drives it caught 4 of 14 collision-course moments (median 1.3 s
early) with 3.0 false alarms per minute, including warm-up at the start of each clip.

## Credits

[flyvis](https://github.com/TuragaLab/flyvis) (Lappalainen et al.), flybrain and the MaleCNS connectome,
the [Waymo Open Dataset](https://waymo.com/open/) (non-commercial license), and
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL).
