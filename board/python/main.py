"""
Recorder for the ankle activity wearable (Linux side).

The sketch sends batches of raw IMU samples over the Bridge. This file writes them to a CSV
when an MQTT command arrives, and reports status back over MQTT.

Command (publish to PREFIX/cmd), plain text:
    label,session,seconds[,delay]      e.g.  walking,s1,60      or  walking,s1,60,8
    stop                               stop the current recording early
Status (subscribe to PREFIX/status):  online / armed / recording / done / error messages.
"""
from arduino.app_utils import *      # first, so it cannot replace the standard imports below

import json
import os
import queue
import re
import threading
import time
import traceback
from collections import Counter, deque

import paho.mqtt.client as mqtt

try:                                           # the recorder still works if these fail to load
    import numpy as np
    from features import CLASSES, CHANNELS, S, W, model1_features, model2_features
    from tinyml import build
except Exception:
    traceback.print_exc()

BROKER = "broker.hivemq.com"
PORT = 1883
PREFIX = "ME193/Kabir/ankle"              # change this if you want a more private topic
CMD_TOPIC = PREFIX + "/cmd"
STATUS_TOPIC = PREFIX + "/status"
LIVE_TOPIC = PREFIX + "/live"             # one JSON message per prediction (every 0.5 s)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_NAMES = ["model1_tree", "model1_knn", "model2_tree", "model2_knn"]

FS = 100                                  # must match the sketch (100 Hz)
ACC_LSB = 2048.0                          # LSB per g at +/-16 g (must match the sketch)
GYR_LSB = 32.8                            # LSB per deg/s at +/-1000 deg/s
LIVE_STEP = 50                            # predict every 0.5 s (50 samples); each window is still 3 s long
DEFAULT_DELAY_S = 5                       # time to get into position before recording starts
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

lock = threading.Lock()
st = {"mode": "idle", "label": "", "session": "", "seconds": 0, "start_at": 0.0,
      "f": None, "t0": None, "n": 0, "path": ""}

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

models = {}                                   # filled by load_models()
buf = deque(maxlen=300)                       # last 3 s of samples while live (units: g and deg/s)
since = {"n": 0}                              # samples since the last prediction
jobs = queue.Queue(maxsize=4)                 # windows waiting for the worker thread
history = {n: deque(maxlen=3) for n in MODEL_NAMES}   # last 3 predictions per model, for smoothing


def load_models():
    with open(os.path.join(MODEL_DIR, "models.json")) as f:
        data = json.load(f)
    for n in MODEL_NAMES:
        models[n] = {"model": build(data[n]), "axis": data["axis"]}
    print("models loaded:", ", ".join(models))


def say(msg):
    print(msg)
    try:
        client.publish(STATUS_TOPIC, msg, qos=1)
    except Exception as e:
        print("status publish failed:", e)


def finish(reason):
    """Close the file and go idle. Call with the lock held."""
    if st["f"] is not None:
        st["f"].close()
    n, path = st["n"], st["path"]
    st.update(mode="idle", f=None, t0=None, n=0)
    say(f"{reason}: {n} samples ({n / FS:.1f} s) -> {os.path.basename(path)}")


def clean(text):
    return re.sub(r"[^A-Za-z0-9_-]", "", text)


def on_connect(c, userdata, flags, reason_code, properties):
    c.subscribe(CMD_TOPIC, qos=1)
    say("online")


def on_message(c, userdata, msg):
    text = msg.payload.decode(errors="ignore").strip()
    with lock:
        if text.lower() == "stop":
            if st["mode"] == "recording":
                finish("stopped")
            elif st["mode"] == "live":
                st["mode"] = "idle"
                say("stopped live")
            else:
                st["mode"] = "idle"
                say("stopped (nothing was running)")
            return
        if text.lower() == "live":
            if st["mode"] != "idle":
                say(f"busy ({st['mode']}), send 'stop' first")
            elif not models:
                say("error: models not loaded (check the models folder and the python console)")
            else:
                buf.clear(); since["n"] = 0
                for h in history.values():
                    h.clear()
                st["mode"] = "live"
                say("live: predictions every 0.5 s on " + LIVE_TOPIC)
            return
        if st["mode"] != "idle":
            say(f"busy ({st['mode']}), send 'stop' first")
            return
        parts = [p.strip() for p in text.split(",")]
        try:
            label, session = clean(parts[0]), clean(parts[1])
            seconds = float(parts[2])
            delay = float(parts[3]) if len(parts) > 3 else DEFAULT_DELAY_S
            if not label or not session or seconds <= 0:
                raise ValueError
        except (IndexError, ValueError):
            say("error: use label,session,seconds[,delay]  e.g. walking,s1,60")
            return
        st.update(mode="armed", label=label, session=session, seconds=seconds,
                  start_at=time.time() + delay)
        say(f"armed: {label} {session} for {seconds:g} s, starting in {delay:g} s")


def open_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    base = f"{st['label']}_{st['session']}"
    path = os.path.join(DATA_DIR, base + ".csv")
    k = 2
    while os.path.exists(path):                 # never overwrite an earlier recording
        path = os.path.join(DATA_DIR, f"{base}_{k}.csv")
        k += 1
    f = open(path, "w")
    f.write("t_ms,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps\n")
    st.update(f=f, path=path, n=0, t0=None, mode="recording")
    say(f"recording {st['label']} {st['session']}")


def handle_batch(batch):
    with lock:
        if st["mode"] == "idle":
            return
        if st["mode"] == "armed":
            if time.time() < st["start_at"]:
                return
            open_file()
        need = int(st["seconds"] * FS)
        for s in batch.split(";"):
            if not s:
                continue
            try:
                t, ax, ay, az, gx, gy, gz = (int(v) for v in s.split(","))
            except ValueError:
                continue
            if st["mode"] == "recording":
                if st["t0"] is None:
                    st["t0"] = t
                st["f"].write(f"{t - st['t0']},{ax / ACC_LSB:.4f},{ay / ACC_LSB:.4f},{az / ACC_LSB:.4f},"
                              f"{gx / GYR_LSB:.2f},{gy / GYR_LSB:.2f},{gz / GYR_LSB:.2f}\n")
                st["n"] += 1
                if st["n"] >= need:
                    finish("done")
                    return
            elif st["mode"] == "live":
                buf.append((ax / ACC_LSB, ay / ACC_LSB, az / ACC_LSB, gx / GYR_LSB, gy / GYR_LSB, gz / GYR_LSB))
                since["n"] += 1
                if since["n"] >= LIVE_STEP and len(buf) == W:
                    since["n"] = 0
                    try:
                        jobs.put_nowait(np.array(buf))
                    except queue.Full:
                        pass                   # the worker is behind; skip this window
        if st["f"] is not None:
            st["f"].flush()


def worker():
    """Runs the models off the Bridge thread so sampling is never held up."""
    while True:
        win = jobs.get()
        try:
            axis = CHANNELS.index(models["model1_tree"]["axis"])
            x1 = [model1_features(win, axis)]
            x2 = [model2_features(win)]
            preds, smooth = {}, {}
            for n in MODEL_NAMES:
                label = CLASSES[int(models[n]["model"].predict(x1 if n.startswith("model1") else x2)[0])]
                history[n].append(label)
                preds[n] = label
                smooth[n] = Counter(history[n]).most_common(1)[0][0]
            msg = {"t": round(time.time(), 2), "peak": round(float(x1[0][0]), 3),
                   "preds": preds, "smooth": smooth,
                   "agree": preds["model1_tree"] == preds["model2_tree"]}
            client.publish(LIVE_TOPIC, json.dumps(msg), qos=0)
        except Exception:
            traceback.print_exc()


batches = 0


def on_imu(batch: str):
    global batches
    batches += 1
    try:
        handle_batch(batch)
    except Exception:
        traceback.print_exc()                   # show the real error instead of failing silently


Bridge.provide("imu", on_imu)

try:
    load_models()
except Exception:
    traceback.print_exc()
    print("live mode is off until the models load; recording still works")
threading.Thread(target=worker, daemon=True).start()

client.on_connect = on_connect
client.on_message = on_message
client.connect_async(BROKER, PORT, 60)
client.loop_start()


last_print = 0.0


def loop():
    global last_print
    time.sleep(0.05)
    if time.time() - last_print >= 2:
        last_print = time.time()
        print(f"batches={batches} mode={st['mode']}")


App.run(user_loop=loop)
