"""
SMART HSRP — FULL PIPELINE TEST
=================================
Tests every detection on every vehicle in the video:
  - ALL detected vehicles (clean + violating) appear in output
  - HSRP label (hsrp / non_hsrp / unknown) per vehicle
  - Helmet status (HELMET / NO_HELMET / UNCERTAIN / -) per vehicle
  - Plate OCR text per vehicle
  - Violation type + confidence where applicable
  - Annotated video: ALL detections drawn (violations=red, clean=green, unclassified=grey)
  - Complete JSON report saved to disk
  - Per-check PASS / FAIL printed at the end

Usage:
    python test_full_pipeline.py [path/to/video.mp4]
    # if no argument given, falls back to VIDEO_PATH below
"""

import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# ── Adjust this if you don't pass an argument ─────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parents[2]
VIDEO_PATH = BASE_DIR / "test_videos" / "test16.mp4"
# ─────────────────────────────────────────────────────────────────────────────

from backend.core.video_pipeline import process_video, generate_violation_summary
from backend.services.video_reader import read_video
from backend.core import rules
from backend.utils.converters import make_json_serializable

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = BASE_DIR / "test_outputs" / "full_pipeline_test"
OUTPUT_VIDEO     = OUTPUT_DIR / "annotated_output.mp4"
OUTPUT_JSON      = OUTPUT_DIR / "full_report.json"
PLATE_CROPS_DIR  = OUTPUT_DIR / "plate_crops"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLATE_CROPS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR SCHEME (BGR)
# ─────────────────────────────────────────────────────────────────────────────
C_VIOLATION = (0,   0,   220)   # Red    – confirmed violation
C_PREDICTED = (0,   200, 220)   # Yellow – predicted / warning
C_CLEAN     = (50,  200, 80)    # Green  – clean / compliant
C_TRACKING  = (160, 160, 160)   # Grey   – tracking, not yet classified
C_PLATE     = (210, 130, 0)     # Blue   – plate box
C_RIDER     = (200, 80,  200)   # Purple – rider / person

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = 0.44
FONT_MED    = 0.55
THICKNESS   = 2
TAG_H       = 18


# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_box(frame, bbox, colour, label="", thickness=THICKNESS):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    if label:
        (tw, _), _ = cv2.getTextSize(label, FONT, FONT_SMALL, 1)
        ty1 = max(y1 - TAG_H, 0)
        cv2.rectangle(frame, (x1, ty1), (x1 + tw + 6, ty1 + TAG_H), colour, -1)
        cv2.putText(frame, label, (x1 + 3, ty1 + TAG_H - 4),
                    FONT, FONT_SMALL, (255, 255, 255), 1, cv2.LINE_AA)


def put_text_shadow(frame, text, pos, scale=0.55, colour=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, pos, FONT, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale, colour, thickness, cv2.LINE_AA)


def fc(conf):
    return f"{conf:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# ASSERTIONS
# ─────────────────────────────────────────────────────────────────────────────
_checks = []   # list of (section, passed, message)

def check(section, condition, msg):
    icon = "✅" if condition else "❌"
    print(f"  {icon}  [{section}] {msg}")
    _checks.append((section, bool(condition), msg))
    return bool(condition)

def section_header(title):
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Allow passing video path as CLI arg
    vpath = Path(sys.argv[1]) if len(sys.argv) > 1 else VIDEO_PATH
    if not vpath.exists():
        print(f"[ERROR] Video not found: {vpath.resolve()}")
        sys.exit(1)

    print("=" * 64)
    print("  SMART HSRP — FULL PIPELINE TEST")
    print(f"  Video : {vpath.name}")
    print(f"  Output: {OUTPUT_DIR.resolve()}")
    print("=" * 64)

    # ── 1. RUN PIPELINE ───────────────────────────────────────────────────────
    section_header("1. RUNNING PIPELINE  (annotate_violations=True, annotate_no_violations=True)")
    t0 = time.time()

    output = process_video(
        video_path=str(vpath),
        output_video_path=None,   # we do our own annotation pass below
        frame_skip=1,
        enable_tracking=True,
        enable_ocr_stabilization=True,
        enable_temporal_fusion=True,
        enable_prediction=True,
        enable_adaptive_thresholds=True,
        enable_db_gating=True,
        annotate_violations=True,
        annotate_no_violations=True,   # draw ALL vehicles
        ocr_mode="always",             # OCR on every frame
    )

    elapsed = time.time() - t0
    print(f"\n  Pipeline finished in {elapsed:.2f}s")

    # ── 2. BASIC OUTPUT SHAPE CHECKS ─────────────────────────────────────────
    section_header("2. OUTPUT STRUCTURE")
    check("output", isinstance(output, dict),             "output is a dict")
    check("output", "frames"          in output,          "'frames' key present")
    check("output", "violations"      in output,          "'violations' key present")
    check("output", "track_summaries" in output,          "'track_summaries' key present")
    check("output", "metadata"        in output,          "'metadata' key present")
    check("output", "temporal_stats"  in output,          "'temporal_stats' key present")
    check("output", len(output["frames"]) > 0,            "at least one frame processed")

    frames    = output["frames"]
    track_sum = output["track_summaries"]
    metadata  = output["metadata"]
    violations = output["violations"]

    # ── 3. METADATA ───────────────────────────────────────────────────────────
    section_header("3. METADATA")
    check("metadata", metadata.get("total_frames_processed", 0) > 0,
          f"frames_processed = {metadata.get('total_frames_processed')}")
    check("metadata", metadata.get("avg_fps", 0) > 0,
          f"avg_fps = {metadata.get('avg_fps')}")
    check("metadata", metadata.get("total_time_seconds", 0) > 0,
          f"total_time_seconds = {metadata.get('total_time_seconds')}")

    # ── 4. TRACK SUMMARIES — ALL VEHICLES ────────────────────────────────────
    section_header("4. TRACK SUMMARIES (should include ALL vehicles, not just violators)")
    check("tracks", len(track_sum) > 0,
          f"track_summaries has {len(track_sum)} tracks")

    # Count violating vs clean tracks
    violating_tracks = [t for t in track_sum.values() if t.get("violation_type")]
    clean_tracks     = [t for t in track_sum.values() if not t.get("violation_type")]
    check("tracks", True,
          f"  → violating tracks : {len(violating_tracks)}")
    check("tracks", True,
          f"  → clean tracks     : {len(clean_tracks)}")
    check("tracks", len(clean_tracks) > 0,
          "at least one clean (non-violating) vehicle in track_summaries")

    # Required fields on every track
    required_fields = ["track_id", "vehicle_class", "first_frame", "last_frame"]
    for tid, t in track_sum.items():
        for f in required_fields:
            check("tracks", f in t,
                  f"track {tid} has field '{f}'")
        break   # just check first track for brevity

    # ── 5. HSRP LABELS ───────────────────────────────────────────────────────
    section_header("5. HSRP LABELS per track")
    hsrp_present   = [t for t in track_sum.values() if t.get("hsrp_label") is not None]
    hsrp_hsrp      = [t for t in hsrp_present if t["hsrp_label"] == "hsrp"]
    hsrp_non       = [t for t in hsrp_present if t["hsrp_label"] == "non_hsrp"]
    check("hsrp", True,
          f"tracks with hsrp_label : {len(hsrp_present)} "
          f"({len(hsrp_hsrp)} HSRP, {len(hsrp_non)} Non-HSRP)")
    # Check confidence populated where label exists
    for t in hsrp_present[:3]:
        check("hsrp", "hsrp_confidence" in t,
              f"track {t['track_id']} has hsrp_confidence={t.get('hsrp_confidence')}")

    # ── 6. HELMET STATUS ─────────────────────────────────────────────────────
    section_header("6. HELMET STATUS per track (motorcycles only)")
    moto_tracks    = [t for t in track_sum.values()
                      if (t.get("vehicle_class") or "").lower() == "motorcycle"]
    helmet_present = [t for t in moto_tracks if t.get("helmet_status") is not None]
    helmet_ok      = [t for t in helmet_present if t["helmet_status"] == "HELMET"]
    helmet_no      = [t for t in helmet_present if t["helmet_status"] == "NO_HELMET"]
    helmet_unc     = [t for t in helmet_present if t["helmet_status"] == "UNCERTAIN"]
    check("helmet", True,
          f"motorcycle tracks : {len(moto_tracks)}, with helmet data : {len(helmet_present)}")
    check("helmet", True,
          f"  → HELMET={len(helmet_ok)}  NO_HELMET={len(helmet_no)}  UNCERTAIN={len(helmet_unc)}")
    for t in track_sum.values():
        vclass = (t.get("vehicle_class") or "").lower()
        if vclass not in ("motorcycle", "bicycle", "bike"):
            check("helmet", t.get("helmet_status") is None or t.get("helmet_status") == "",
                  f"4-wheeler track {t['track_id']} has no helmet_status (correct)")
            break

    # ── 7. OCR / PLATE NUMBERS ───────────────────────────────────────────────
    section_header("7. OCR / PLATE NUMBERS")
    plates_found = [t for t in track_sum.values() if t.get("plate_number")]
    check("ocr", True,
          f"tracks with plate_number : {len(plates_found)}")
    for t in plates_found[:5]:
        check("ocr", len(t["plate_number"]) >= 4,
              f"  track {t['track_id']} plate='{t['plate_number']}'  conf={t.get('ocr_confidence', 0):.2f}")

    # ── 8. VIOLATIONS ─────────────────────────────────────────────────────────
    section_header("8. VIOLATION RECORDS")
    check("violations", isinstance(violations, list),
          f"violations is a list with {len(violations)} entries")
    for i, v in enumerate(violations):
        check("violations", "violation_type"   in v, f"violation[{i}] has violation_type={v.get('violation_type')}")
        check("violations", "track_id"         in v, f"violation[{i}] has track_id={v.get('track_id')}")
        check("violations", "vehicle_class"    in v, f"violation[{i}] has vehicle_class={v.get('vehicle_class')}")
        check("violations", "quality_score"    in v, f"violation[{i}] quality_score={v.get('quality_score', 0):.2f}")

    # ── 9. VIOLATION SUMMARY ──────────────────────────────────────────────────
    section_header("9. VIOLATION SUMMARY")
    summary = generate_violation_summary(output)
    check("summary", "total"        in summary, f"total violations = {summary.get('total')}")
    check("summary", "by_type"      in summary, f"by_type keys = {list(summary.get('by_type', {}).keys())}")
    check("summary", "needs_review" in summary, f"needs_review = {summary.get('needs_review')}")
    check("summary", "auto_store"   in summary, f"auto_store   = {summary.get('auto_store')}")
    for vtype, stats in summary.get("by_type", {}).items():
        check("summary", stats.get("count", 0) > 0,
              f"  {vtype}: count={stats['count']}  avg_conf={stats['avg_conf']:.2f}")

    # ── 10. TEMPORAL STATS ────────────────────────────────────────────────────
    section_header("10. TEMPORAL FUSION STATS")
    ts = output.get("temporal_stats", {})
    for key in ("hsrp_fusion", "helmet_fusion", "ocr_fusion"):
        check("temporal", key in ts, f"'{key}' present in temporal_stats")
        if key in ts:
            gstats = ts[key].get("global_stats", {})
            check("temporal", gstats.get("total_decisions", 0) > 0,
                  f"  {key}: total_decisions={gstats.get('total_decisions')}")

    # ── 11. PER-FRAME STRUCTURE ───────────────────────────────────────────────
    section_header("11. PER-FRAME STRUCTURE (spot check 5 frames)")
    frame_map = {f["frame_id"]: f for f in frames}
    sample_frames = list(frame_map.values())[:5]
    for fd in sample_frames:
        fid = fd["frame_id"]
        check("frames", "entities"     in fd, f"frame {fid} has 'entities'")
        check("frames", "associations" in fd, f"frame {fid} has 'associations'")
        check("frames", "violations"   in fd, f"frame {fid} has 'violations'")

    # ── 12. GENERATE ANNOTATED VIDEO ─────────────────────────────────────────
    section_header("12. GENERATING ANNOTATED VIDEO  (ALL detections)")
    print(f"  Writing to: {OUTPUT_VIDEO}")
    writer = None
    frames_written = 0
    plate_crops_saved = 0

    for frame_id, frame in read_video(str(vpath)):
        if frame_id not in frame_map:
            continue

        fd         = frame_map[frame_id]
        vehicles   = fd.get("entities", {}).get("vehicles", [])
        plates     = fd.get("entities", {}).get("plates",   [])
        assoc_vp   = fd.get("associations", {}).get("vehicle_plate", {})
        assoc_vr   = fd.get("associations", {}).get("vehicle_rider", {})
        enf_list   = fd.get("enforcements", {}).get("two_wheelers", [])
        enf_by_vid = {e["vehicle_id"]: e for e in enf_list}
        preds      = fd.get("predictions", {})
        viol_vids  = {v["vehicle_id"] for v in fd.get("violations", [])}

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(OUTPUT_VIDEO),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25, (w, h),
            )

        # Frame number overlay
        put_text_shadow(frame, f"Frame {frame_id}", (10, 28), scale=0.7)

        # Stats overlay (top right)
        n_veh   = len(vehicles)
        n_viol  = len(fd.get("violations", []))
        put_text_shadow(frame,
            f"Vehicles:{n_veh}  Violations:{n_viol}",
            (10, 54), scale=0.5, colour=(0, 200, 220))

        drawn_riders = set()

        for v in vehicles:
            vid      = v.get("id", "")
            track_id = v.get("track_id", vid)
            vclass   = (v.get("class") or v.get("vehicle_class") or "vehicle").lower()
            bbox     = v.get("bbox", [0, 0, 0, 0])
            vconf    = v.get("confidence", 0.0)

            pid   = assoc_vp.get(vid)
            plate = next((p for p in plates if p["id"] == pid), None) if pid else None
            enf   = enf_by_vid.get(vid)
            pred  = preds.get(track_id, {})

            # ── Classify this vehicle ────────────────────────────────────
            hsrp_label    = plate.get("hsrp")               if plate else None
            hsrp_conf     = plate.get("hsrp_confidence", 0) if plate else 0.0
            helmet_info   = enf.get("helmet") if (enf and enf.get("helmet")) else None
            helmet_status = helmet_info.get("status") if helmet_info else None
            helmet_conf   = helmet_info.get("confidence", 0.0) if helmet_info else 0.0
            is_helmet_viol = bool(helmet_info and helmet_info.get("is_violation", False))

            # Use rules for official violation decision
            hsrp_is_violation = rules.hsrp_violation(
                is_hsrp=(hsrp_label == "hsrp"),
                confidence=hsrp_conf,
            ) if hsrp_label is not None else False

            helmet_is_violation = rules.helmet_violation(
                helmet_detected=(helmet_status == "HELMET"),
                confidence=helmet_conf,
            ) if helmet_status is not None else False

            is_confirmed_viol = (vid in viol_vids
                                 or hsrp_is_violation
                                 or helmet_is_violation)
            is_predicted = (pred.get("any_warning", False)
                            or pred.get("any_confirmed", False))

            # ── Colour ───────────────────────────────────────────────────
            if is_confirmed_viol:
                colour = C_VIOLATION
            elif is_predicted:
                colour = C_PREDICTED
            elif hsrp_label is not None or helmet_status is not None:
                colour = C_CLEAN
            else:
                colour = C_TRACKING

            # ── Vehicle label ────────────────────────────────────────────
            parts = [f"{vclass} {track_id}"]
            if hsrp_label == "hsrp":
                parts.append(f"HSRP {fc(hsrp_conf)}")
            elif hsrp_label == "non_hsrp":
                parts.append(f"Non-HSRP {fc(hsrp_conf)}")
            if helmet_status == "HELMET":
                parts.append(f"Helmet {fc(helmet_conf)}")
            elif helmet_status == "NO_HELMET":
                parts.append(f"No Helmet {fc(helmet_conf)}")
            elif helmet_status == "UNCERTAIN":
                parts.append("Uncertain")
            # Predictor risk
            hsrp_risk   = pred.get("hsrp",   {}).get("risk_score", 0.0)
            helmet_risk = pred.get("helmet", {}).get("risk_score", 0.0)
            if hsrp_risk > 0.3:
                parts.append(f"[H-risk {fc(hsrp_risk)}]")
            if helmet_risk > 0.3:
                parts.append(f"[R-risk {fc(helmet_risk)}]")

            draw_box(frame, bbox, colour, "  |  ".join(parts))

            # ── Plate box + OCR ──────────────────────────────────────────
            if plate:
                pbbox = plate.get("bbox")
                if pbbox and len(pbbox) == 4:
                    ocr_text = plate.get("ocr_text") or ""
                    p_label  = ocr_text if ocr_text else "Plate"
                    draw_box(frame, pbbox, C_PLATE, p_label, thickness=1)

                    # Save plate crop
                    px1, py1, px2, py2 = [int(x) for x in pbbox]
                    crop = frame[py1:py2, px1:px2]
                    if crop.size > 0:
                        crop_path = PLATE_CROPS_DIR / f"f{frame_id}_{track_id}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        plate_crops_saved += 1

            # ── Rider box ────────────────────────────────────────────────
            if enf and enf.get("rider_found"):
                rbbox = enf.get("person_bbox")
                if rbbox:
                    rider_id = assoc_vr.get(vid, "")
                    if rider_id not in drawn_riders:
                        drawn_riders.add(rider_id)
                        r_colour = (C_VIOLATION if is_helmet_viol
                                    else C_CLEAN if helmet_status == "HELMET"
                                    else C_TRACKING)
                        draw_box(frame, rbbox, r_colour, "Rider", thickness=1)

        writer.write(frame)
        frames_written += 1

    if writer:
        writer.release()

    check("video", OUTPUT_VIDEO.exists(),
          f"annotated video written ({frames_written} frames)")
    check("video", plate_crops_saved > 0,
          f"plate crops saved: {plate_crops_saved}")

    # ── 13. SAVE JSON REPORT ──────────────────────────────────────────────────
    section_header("13. SAVING JSON REPORT")

    # Build complete report
    report = {
        "status":   "completed",
        "metadata": output.get("metadata", {}),
        "summary":  summary,
        "track_summaries": {
            tid: {
                "track_id":            t.get("track_id"),
                "vehicle_class":       t.get("vehicle_class"),
                "first_frame":         t.get("first_frame"),
                "last_frame":          t.get("last_frame"),
                "hsrp_label":          t.get("hsrp_label"),
                "hsrp_confidence":     round(t.get("hsrp_confidence", 0.0), 4),
                "helmet_status":       t.get("helmet_status"),
                "helmet_confidence":   round(t.get("helmet_confidence", 0.0), 4),
                "plate_number":        t.get("plate_number"),
                "ocr_confidence":      round(t.get("ocr_confidence", 0.0), 4),
                "violation_type":      t.get("violation_type"),
                "violation_confidence": round(t.get("violation_confidence", 0.0), 4),
                "quality_score":       round(t.get("quality_score", 0.0), 4),
                "should_store":        t.get("should_store", False),
                "needs_review":        t.get("needs_review", False),
                "prediction_preceded": t.get("prediction_preceded", False),
            }
            for tid, t in track_sum.items()
        },
        "violations":        violations,
        "temporal_stats":    output.get("temporal_stats", {}),
        "adaptive_thresholds": output.get("adaptive_thresholds", {}),
        "output_video":      str(OUTPUT_VIDEO),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(report), f, indent=2)

    check("json", OUTPUT_JSON.exists(),
          f"JSON report saved ({OUTPUT_JSON.stat().st_size // 1024} KB)")

    # Quick sanity: re-read JSON to confirm it's valid
    with open(OUTPUT_JSON) as f:
        loaded = json.load(f)
    check("json", loaded.get("status") == "completed",
          "JSON round-trip: status == 'completed'")
    check("json", len(loaded.get("track_summaries", {})) == len(track_sum),
          f"JSON round-trip: {len(loaded['track_summaries'])} track_summaries match")

    # ── 14. CONSOLE SUMMARY TABLE ─────────────────────────────────────────────
    section_header("14. TRACK-LEVEL DETAIL TABLE")
    col = "{:<14} {:<12} {:<12} {:<14} {:<12} {:<12} {:<22} {:<8}"
    print(col.format(
        "Track ID", "Vehicle", "HSRP", "HSRP Conf",
        "Helmet", "Hlmt Conf", "Violation", "Stored"
    ))
    print("─" * 108)
    for tid, t in sorted(track_sum.items(), key=lambda x: x[1].get("first_frame", 0)):
        vclass = (t.get("vehicle_class") or "?").lower()
        is_two = vclass in ("motorcycle", "bicycle", "bike")

        hsrp_str   = t.get("hsrp_label") or "—"
        hsrp_c     = f"{t.get('hsrp_confidence', 0):.2f}"
        helm_str   = t.get("helmet_status") or ("—" if not is_two else "n/a")
        helm_c     = f"{t.get('helmet_confidence', 0):.2f}" if is_two else "—"
        vtype      = t.get("violation_type") or "✅ clean"
        stored     = "YES" if t.get("should_store") else "—"

        print(col.format(
            tid[:14], vclass[:12],
            hsrp_str[:12], hsrp_c,
            helm_str[:12], helm_c,
            vtype[:22], stored
        ))

    # ── FINAL PASS / FAIL SUMMARY ─────────────────────────────────────────────
    section_header("FINAL TEST RESULTS")
    passed = sum(1 for _, p, _ in _checks if p)
    failed = sum(1 for _, p, _ in _checks if not p)
    total  = len(_checks)

    if failed:
        print("\n  FAILED CHECKS:")
        for sec, p, msg in _checks:
            if not p:
                print(f"    ❌  [{sec}] {msg}")

    print(f"\n  {'─'*40}")
    print(f"  Passed : {passed}/{total}")
    print(f"  Failed : {failed}/{total}")
    print(f"  {'─'*40}")

    print(f"\n  Output files:")
    print(f"    🎬 Video : {OUTPUT_VIDEO.resolve()}")
    print(f"    📄 JSON  : {OUTPUT_JSON.resolve()}")
    print(f"    🖼️  Crops : {PLATE_CROPS_DIR.resolve()} ({plate_crops_saved} crops)")
    print()

    if failed:
        print("  ❌  SOME CHECKS FAILED — see above")
        sys.exit(1)
    else:
        print("  ✅  ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()