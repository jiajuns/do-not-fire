"""Summarize real-robot ROS 2 bags without installing ROS.

This first-stage parser extracts odometry travel, final known-map area, coverage
curves, and map-gate events.  Cross-robot overlap must be computed after maps are
transformed into one verified global frame; this script deliberately does not
pretend that identically named local ``map`` frames are aligned.

Example:
  python code/parse_real_bags.py \
    --bag robot0=~/svoi_runs/sync_001/robot0/bag \
    --bag robot1=~/svoi_runs/sync_001/robot1/bag \
    --output results/real/sync_001_summary.json
"""
import argparse
import json
import math
from pathlib import Path


ODOM_TYPES = {"nav_msgs/msg/Odometry"}
MAP_TYPES = {"nav_msgs/msg/OccupancyGrid"}
EVENT_TOPICS = {"/svoi/comm_events"}


def _stamp(msg):
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def _known_map_summary(msg):
    known = sum(1 for value in msg.data if int(value) >= 0)
    resolution = float(msg.info.resolution)
    return {
        "stamp": _stamp(msg),
        "frame_id": str(msg.header.frame_id),
        "width": int(msg.info.width),
        "height": int(msg.info.height),
        "resolution_m": resolution,
        "known_cells": known,
        "known_area_m2": known * resolution * resolution,
    }


def summarize_bag(path):
    try:
        from rosbags.highlevel import AnyReader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pip install rosbags") from exc

    bag_path = Path(path).expanduser().resolve()
    if not bag_path.exists():
        raise FileNotFoundError(bag_path)

    odom = {}
    maps = {}
    gate = {
        "published": 0, "missed": 0, "serialized_map_bytes": 0,
        "event_times": [], "source_ages_sec": [],
    }
    # Humble sqlite bags often omit embedded message definitions. All topics
    # parsed here are standard messages, so use the matching built-in store.
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([bag_path], default_typestore=typestore) as reader:
        connections = [
            c for c in reader.connections
            if c.msgtype in ODOM_TYPES | MAP_TYPES or c.topic in EVENT_TOPICS
        ]
        for connection, bag_stamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            if connection.msgtype in ODOM_TYPES:
                state = odom.setdefault(connection.topic, {
                    "samples": 0, "travel_m": 0.0, "previous": None,
                    "first_bag_time": bag_stamp_ns * 1e-9,
                })
                pos = msg.pose.pose.position
                # Ground-robot travel is planar; ignore vertical LIO noise.
                current = (float(pos.x), float(pos.y))
                if state["previous"] is not None:
                    state["travel_m"] += math.dist(state["previous"], current)
                state["previous"] = current
                state["samples"] += 1
                state["last_bag_time"] = bag_stamp_ns * 1e-9
            elif connection.msgtype in MAP_TYPES:
                state = maps.setdefault(connection.topic, {"samples": 0, "coverage_curve": []})
                summary = _known_map_summary(msg)
                state["samples"] += 1
                state["final"] = summary
                state["coverage_curve"].append([
                    bag_stamp_ns * 1e-9, summary["known_cells"], summary["known_area_m2"]
                ])
            elif connection.topic in EVENT_TOPICS:
                data = list(msg.data)
                if len(data) < 7:
                    continue
                event_type, event_time, source_time = int(data[0]), float(data[3]), float(data[4])
                if event_type == 1:
                    gate["published"] += 1
                    gate["serialized_map_bytes"] += int(data[5])
                    gate["event_times"].append(event_time)
                    if source_time > 0.0:
                        gate["source_ages_sec"].append(event_time - source_time)
                else:
                    gate["missed"] += 1

    for state in odom.values():
        state.pop("previous", None)
        if "last_bag_time" in state:
            state["duration_sec"] = state["last_bag_time"] - state["first_bag_time"]
    times = gate.pop("event_times")
    ages = gate.pop("source_ages_sec")
    gate["mean_interval_sec"] = (
        sum(b - a for a, b in zip(times, times[1:])) / (len(times) - 1)
        if len(times) > 1 else None
    )
    gate["mean_source_age_sec"] = sum(ages) / len(ages) if ages else None
    return {"bag": str(bag_path), "odometry": odom, "maps": maps, "gate": gate}


def _parse_bag_arg(value):
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path = value
        label = Path(path).name
    if not label or not path:
        raise argparse.ArgumentTypeError("use LABEL=PATH or PATH")
    return label, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", action="append", required=True, type=_parse_bag_arg)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {label: summarize_bag(path) for label, path in args.bag}
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
