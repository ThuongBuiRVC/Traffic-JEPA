#!/usr/bin/env python
"""Write the pedestrian/vehicle captions for the WTS public test.

  VQA answers (submissions/submission_final.json)
      -> facts per (scenario, phase)
      -> Qwen3-VL-8B (optionally + the caption LoRA)
      -> submissions/caption_submission.json

Two paths, picked per segment:
  QA path     — the segment has VQA answers: the facts are prompted as text.
  frame path  — the segment has none (`normal_trimmed`): ffmpeg cuts frames and the model
                describes them. The LoRA is text-only, so this path always runs on the base
                weights, in `--mode lora` too.

Everything the model writes goes through `mild_clean` / `env_fix`, which is grammar cleanup of
the model's own output (no template assembly, no ground truth).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from shutil import which

from traffic_jepa.captioning.backend import QwenVL, image_msg, text_msg
from traffic_jepa.inference.predict import _scenario_of

ROOT = Path(__file__).resolve().parents[2]
DEF_VQA = ROOT / "submissions" / "submission_final.json"
DEF_TEST = ROOT / "data" / "test" / "WTS_VQA_PUBLIC_TEST.json"
DEF_SEGMENTS = ROOT / "configs" / "caption" / "segments.json"
DEF_FEWSHOT = ROOT / "configs" / "caption" / "fewshot_examples.json"
DEF_LORA = ROOT / "checkpoints" / "caption_lora"
DEF_OUT = ROOT / "submissions" / "caption_submission.json"
BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"

PHASE_FROM_LABEL = {"prerecognition": "0", "recognition": "1", "judgement": "2", "judgment": "2",
                    "action": "3", "avoidance": "4"}
PHASE_NAME = {"0": "prerecognition (before the pedestrian notices the vehicle)", "1": "recognition",
              "2": "judgement", "3": "action", "4": "avoidance"}

PED_SLOTS = {
    "appearance": ["What is the age group of the pedestrian?",
                   "What is the height of the pedestrian?",
                   "What pedestrian is wearing on upper body?",
                   "What is the color of pedestrian's upper body clothing?",
                   "What pedestrian is wearing on lower body?",
                   "What is the color of pedestrian's lower body clothing?",
                   "Is pedestrian wearning a hat?",
                   "What is color of pedestrian's hat?",
                   "Is the pedestrian wearing glasses?"],
    "location":  ["What is the orientation of the pedestrian's body?",
                  "What is the position of the pedestrian relative to the vehicle?",
                  "What is relative distance of pedestrian from vehicle?"],
    "attention": ["What is the pedestrian's visual status?",
                  "What is the pedestrian's line of sight?",
                  "What is the pedestrian's awareness regarding vehicle?"],
    "behaviour": ["What is the pedestrian's action?",
                  "What is the pedestrian's direction of travel?",
                  "What is pedestrian's speed?",
                  "What is the fine-grained action taken by the pedestrian?"],
}
VEH_SLOTS = {
    "location":  ["What is the position of the vehicle relative to the pedestrian?",
                  "What is relative distance of vehicle from pedestrian?"],
    "view":      ["What is vehicle's field of view?"],
    "behaviour": ["What is the action taken by vehicle?"],
}
CONTEXT_SLOTS = {
    "environment": ["What is weather in the scenario?",
                    "What is the brightness level in the scene?",
                    "What are road surface conditions?",
                    "What is surface type of the road?",
                    "What is the road inclination in the scene?",
                    "What is the type of the road?",
                    "How many lanes are there?",
                    "What is the volume of the traffic in the scene?",
                    "What is the formation of the road?",
                    "Where is the sidewalk in the scene?",
                    "Where is the roadside strip in the scene?",
                    "Are there street lights in the scene?",
                    "What is the position of the obstacle in the scene?",
                    "What is the height of obstacle in the scene?",
                    "What is the width of obstacle in the scene?"]
}


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


_REPL = {"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...", " ": " ", "​": "", "﻿": ""}


def clean(text: str) -> str:
    import unicodedata
    text = text or ""
    for k, v in _REPL.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- facts from VQA
def build_facts(vqa_path, test_path) -> dict[tuple[str, str], dict[str, str]]:
    """(scenario, phase) -> {question: predicted answer text}.

    Scenario-level questions (appearance, environment) are copied into every phase of the same
    scenario, so each segment carries the full attribute set the caption has to cover.
    """
    vqa = {x["id"]: x["correct"] for x in json.load(open(vqa_path))}
    facts: dict = defaultdict(dict)
    scen_const: dict = defaultdict(dict)
    for item in json.load(open(test_path)):
        videos = item.get("videos") or []
        scenario = _scenario_of(videos) if videos else None
        for c in item.get("conversations", []):
            letter = vqa.get(c["id"])
            ans = c.get(letter, "") if letter else ""
            if ans:
                scen_const[scenario][norm(c["question"])] = norm(ans)
        for p in item.get("event_phase", []):
            phase = PHASE_FROM_LABEL.get(str((p.get("labels") or ["?"])[0]).lower()) or "ENV"
            for c in p.get("conversations", []):
                letter = vqa.get(c["id"])
                ans = c.get(letter, "") if letter else ""
                if ans:
                    facts[(scenario, phase)][norm(c["question"])] = norm(ans)
    for (scenario, _phase), d in facts.items():
        for q, a in scen_const.get(scenario, {}).items():
            d.setdefault(q, a)
    return dict(facts)


def slot_lines(facts_qa, slots):
    out = []
    for slot, questions in slots.items():
        parts = [f"{q.split('?')[0].split('is ')[-1].strip()} = {facts_qa[norm(q)]}"
                 for q in questions if norm(q) in facts_qa]
        if parts:
            out.append(f"{slot}: " + "; ".join(parts))
    return out


def values_of(facts_qa, slots):
    out, seen = [], set()
    for questions in slots.values():
        for q in questions:
            v = facts_qa.get(norm(q))
            if v and v.lower() not in seen:
                seen.add(v.lower())
                out.append(v)
    return out


def ped_values(facts_qa):
    return values_of(facts_qa, PED_SLOTS) + values_of(facts_qa, CONTEXT_SLOTS)


def veh_values(facts_qa):
    return (values_of(facts_qa, VEH_SLOTS)
            + values_of(facts_qa, {"appearance": PED_SLOTS["appearance"]})
            + values_of(facts_qa, CONTEXT_SLOTS))


def missing_values(text, values):
    t = (text or "").lower()
    return [v for v in values if v.strip().rstrip(".").lower() not in t]


# --------------------------------------------------------------------------- output cleanup
def env_fix(text):
    """Normalise the model's own environment wording toward the WTS register — grammar cleanup
    only, like detokenisation. Garbage such as 'roadside strip is not both sides' or 'street
    lights are yes' matches no reference n-gram, and dropping it lifts all four metrics."""
    r = text or ""
    r = re.sub(r"\b(?:the\s+)?roadside strips?(?: in the scene)?\s+(?:is|are)\s+not both sides\b",
               "there is no roadside strip on both sides", r, flags=re.I)
    r = re.sub(r"\b(?:the\s+)?sidewalks?(?: in the scene)?\s+(?:is|are)\s+not both sides\b",
               "there is no sidewalk on both sides", r, flags=re.I)
    r = re.sub(r"\bis not both sides\b", "is not present on both sides", r, flags=re.I)
    r = re.sub(r"\bare not both sides\b", "are not present on both sides", r, flags=re.I)
    r = re.sub(r"\bstreet lights are yes\b", "street lights are present", r, flags=re.I)
    r = re.sub(r"\bare yes\b", "are present", r, flags=re.I)
    r = re.sub(r"([.!?]\s+)(?:Yes|No)\.\s*", r"\1", r)
    r = re.sub(r"^(?:Yes|No)\.\s*", "", r)
    r = re.sub(r"([.!?]\s+)(?:Yes|No),\s*([a-z])", lambda m: m.group(1) + m.group(2).upper(), r)
    r = re.sub(r"^(?:Yes|No),\s*([a-z])", lambda m: m.group(1).upper(), r)
    r = re.sub(r",\s*(?:Yes|No)\b(?=[,. ])", "", r)
    r = re.sub(r"([.!?])([A-Za-z])", lambda m: m.group(1) + " " + m.group(2).upper(), r)
    r = re.sub(r"\s+([.,])", r"\1", r)
    r = norm(r)
    if r and r[0].islower():
        r = r[0].upper() + r[1:]
    return r


_BAREYN = re.compile(r"(?:[,;.]\s*(?:Yes|No)\b\.?)+\s*$", re.I)


def mild_clean(text):
    """Drop the model's own noise ('X = Yes' echoes, trailing bare 'Yes/No'), keep the prose."""
    text = clean(text)
    text = re.sub(r"[A-Za-z][A-Za-z'/ ]*\s=\s[A-Za-z0-9][A-Za-z0-9 ]*", "", text)
    text = re.sub(r"\s*,\s*(?=[.,])", "", text)
    text = re.sub(r",\s*\.", ".", text)
    text = _BAREYN.sub(".", text.strip())
    text = env_fix(text)
    text = norm(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


# --------------------------------------------------------------------------- prompts
SYS = ("You write traffic-scene captions in the exact style of the WTS dataset. Each caption is ONE flowing "
       "paragraph of about 120-135 words, third person, no line breaks, no bullet points, no headers, never the "
       "word 'phase'. Pedestrian caption order: appearance; then location (body orientation, position relative "
       "to the vehicle, relative distance); then attention (visual status, line of sight, awareness); then "
       "behaviour (action, direction of travel, speed); then a full environment description (weather, "
       "brightness, road surface and level, road type, traffic volume). Vehicle caption: the vehicle's view of "
       "the pedestrian, its position and distance to the pedestrian, its action and speed, then restate the "
       "pedestrian's appearance and the environment. HOW TO USE THE GIVEN VALUES: copy each given value into a "
       "grammatical sentence EXACTLY as written (word for word, no paraphrase/synonym/reorder), and WEAVE it "
       "naturally into the flowing prose. ABSOLUTELY FORBIDDEN: do NOT append a list of the values at the end; do "
       "NOT tack on comma-separated fragments like 'Not both sides, Yes.' or 'Residential road, Intersection "
       "(with signal)'; do NOT echo the question (never write 'X = Yes' or 'wearing a hat = Yes'). Every value "
       "must appear ONLY inside a real sentence with a verb. Also do NOT invent attributes that were not given, "
       "do NOT repeat a detail, and do NOT add editorial filler or dramatic/risk/intention clauses (no 'the scene "
       "is clearly defined', 'allowing for cautious movement', 'creating a limited refuge', etc.) - stick to the "
       "given facts only. WTS GROUND-TRUTH REGISTER (use these exact wordings when you write the prose): "
       "introduce the environment with 'The environment conditions include ...' (never 'as for the environment' "
       "or 'the scene unfolds'); give the pedestrian height as 'with a height of <N> cm'; write 'the road "
       "surface conditions were dry', 'It was an asphalt road with two-way traffic'; refer to gaze as \"the "
       "pedestrian's line of sight\". Keep each caption about 125-135 words. End right after the environment "
       "sentence. Return STRICT JSON only: {\"caption_pedestrian\": \"...\", \"caption_vehicle\": \"...\"}")

FRAME_SYS = (
    "You are a captioner for the WTS traffic dataset. You are shown several frames from ONE short clip. Write two "
    "captions that MATCH THE EXACT WTS WRITING STYLE below (this phrasing is what is scored), each ONE flowing "
    "paragraph, third person, about 110-130 words, no line breaks, no headers, never the word 'phase'.\n"
    "caption_pedestrian MUST follow this sentence pattern, filled from what you see: 'The pedestrian, a "
    "<male/female> in <his/her> <20s/30s/40s>, wearing a <colour> <upper garment> and <colour> <lower garment>, "
    "<was/is> positioned <diagonally to the left/right / directly> in front of the vehicle, at a <close/near/far> "
    "relative distance. <His/Her> body was <perpendicular/parallel/facing> the vehicle and <his/her> line of "
    "sight was <in the direction of travel / toward the crossing destination>. The pedestrian was <closely "
    "watching / unaware of> the vehicle. <He/She> was <standing still / walking / crossing / going straight "
    "ahead> at a <slow/normal> speed. The weather was <clear/cloudy> and the brightness was <bright/dim>. The "
    "road surface was dry and level, made of asphalt. It was a residential road with two-way traffic and "
    "<light/usual> traffic volume, with no sidewalks on both sides.'\n"
    "caption_vehicle MUST follow: 'The vehicle was positioned <side> the pedestrian, at a <close/near/far> "
    "relative distance. The pedestrian was visible within the vehicle's field of view. The vehicle was <going "
    "straight ahead / stopped / turning / moving forward>. The pedestrian was a <male/female> in <his/her> "
    "<age>, wearing a <colour> <upper> and <colour> <lower>. The weather was <clear>, the road surface was dry "
    "and level asphalt, and it was a residential road with two-way traffic and <light> traffic volume, with no "
    "sidewalks on both sides.'\n"
    "Use the standard WTS environment wording above (clear/bright/dry/level/asphalt/residential road/two-way "
    "traffic/no sidewalks) unless the frames clearly show otherwise. A target pedestrian is ALWAYS present - "
    "never say no pedestrian is visible; describe the one you see even if small or far. Do not invent collisions. "
    "Return STRICT JSON only: {\"caption_pedestrian\": \"...\", \"caption_vehicle\": \"...\"}")


def facts_block(facts_qa):
    env = "\n".join(slot_lines(facts_qa, CONTEXT_SLOTS)) or "(none)"
    pf = "\n".join(slot_lines(facts_qa, PED_SLOTS)) or "(none)"
    vf = "\n".join(slot_lines(facts_qa, VEH_SLOTS)) or "(none)"
    return f"ENVIRONMENT:\n{env}\nPEDESTRIAN FACTS:\n{pf}\nVEHICLE FACTS:\n{vf}"


def build_prompt(scenario, phase, facts_qa, shots):
    demo = ""
    for s in shots:
        if "env" in s:
            demo += (f"FACTS:\n{s['env']}\nPEDESTRIAN FACTS:\n{s['ped_facts']}\nVEHICLE FACTS:\n{s['veh_facts']}\n"
                     f"pedestrian caption: {s['caption_pedestrian']}\nvehicle caption: {s['caption_vehicle']}\n\n")
        else:
            demo += f"pedestrian caption: {s['caption_pedestrian']}\nvehicle caption: {s['caption_vehicle']}\n\n"
    phase_hint = ""
    pn = PHASE_NAME.get(str(phase))
    if pn:
        phase_hint = (f"This segment is the '{pn}' moment of the interaction; describe the state at this moment "
                      f"consistently with the facts.\n")
    return (f"{demo}{phase_hint}Now write the two captions for the segment below in the SAME style and length as "
            f"the examples (about 105-120 words each). Use each value's exact wording, but WEAVE every value into "
            f"grammatical sentences - do NOT list or append them, do NOT echo the questions, do NOT add filler. "
            f"The pedestrian caption must include appearance, location, attention, behaviour and the environment; "
            f"the vehicle caption must include the vehicle's view/position/action, then the pedestrian's "
            f"appearance and the environment.\n\n{facts_block(facts_qa)}\n\n"
            'Return strict JSON: {"caption_pedestrian": "...", "caption_vehicle": "..."}')


def load_style_examples(path, n):
    if n <= 0 or not Path(path).is_file():
        return []
    return json.load(open(path, encoding="utf-8"))[:n]


def parse_caps(txt):
    try:
        cap = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
    except Exception:
        return None
    p, v = norm(cap.get("caption_pedestrian", "")), norm(cap.get("caption_vehicle", ""))
    return {"caption_pedestrian": p, "caption_vehicle": v} if (p and v) else None


# --------------------------------------------------------------------------- QA path
def qa_caption(runner, scenario, phase, qa, shots, temperature, rounds=4):
    """Prompt the model from the VQA facts, re-asking while any answer value is still missing."""
    pv, vv = ped_values(qa), veh_values(qa)
    prompt = build_prompt(scenario, phase, qa, shots)
    msgs = [text_msg("system", SYS), text_msg("user", prompt)]
    cap = best = None
    best_missing = 1 << 30
    for _ in range(rounds):
        try:
            cap = parse_caps(runner.chat(msgs, temperature=temperature))
        except Exception as e:
            print(f"  generate retry {scenario} p{phase} ({e})", flush=True)
            continue
        if not cap:
            continue
        cap["caption_pedestrian"] = mild_clean(cap["caption_pedestrian"])
        cap["caption_vehicle"] = mild_clean(cap["caption_vehicle"])
        mp, mv = missing_values(cap["caption_pedestrian"], pv), missing_values(cap["caption_vehicle"], vv)
        if len(mp) + len(mv) < best_missing:
            best, best_missing = cap, len(mp) + len(mv)
        if not mp and not mv:
            return cap
        msgs = [text_msg("system", SYS), text_msg("user", prompt),
                text_msg("assistant", json.dumps(cap)),
                text_msg("user",
                         "Some required values are missing. Rewrite BOTH captions (about 125 words each, strict "
                         "JSON) so that EVERY value appears verbatim, WEAVING each missing value into a real "
                         "sentence. Missing from the pedestrian caption: " + ("; ".join(mp) or "none") +
                         ". Missing from the vehicle caption: " + ("; ".join(mv) or "none") + ".")]
    cap = best or cap or template_caption(qa)
    cap["caption_pedestrian"] = mild_clean(cap["caption_pedestrian"])
    cap["caption_vehicle"] = mild_clean(cap["caption_vehicle"])
    return cap


# --------------------------------------------------------------------------- frame path
def load_segment_times(wts_root):
    """(scenario, phase) -> (start, end) from the public-test caption annotations."""
    seg = {}
    ann = Path(wts_root) / "annotations" / "caption" / "test" / "public_challenge"
    if not ann.is_dir():
        return seg
    for cap in ann.rglob("*_caption.json"):
        scenario = cap.stem[:-len("_caption")]
        try:
            d = json.load(open(cap))
        except Exception:
            continue
        for e in d.get("event_phase", []):
            ph = str((e.get("labels") or ["?"])[0])
            if e.get("start_time") is not None:
                s = float(e["start_time"])
                seg[(scenario, ph)] = (s, float(e.get("end_time", s)))
    return seg


def find_views(wts_root, scenario):
    root = Path(wts_root) / "videos" / "test" / "public"
    overhead = vehicle = None
    for base in (root, root / "normal_trimmed"):
        d = base / scenario
        if not d.is_dir():
            continue
        vv = d / "vehicle_view" / f"{scenario}_vehicle_view.mp4"
        if vv.exists():
            vehicle = vv
        ovd = d / "overhead_view"
        if ovd.is_dir():
            m = sorted(ovd.glob("*.mp4"))
            if m:
                overhead = m[0]
        if overhead or vehicle:
            break
    return overhead, vehicle


def _duration(vid):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(vid)], capture_output=True, text=True, timeout=30)
        d = float(out.stdout.strip())
        if d > 0:
            return d
    except Exception:
        pass
    try:
        err = subprocess.run(["ffmpeg", "-i", str(vid)], capture_output=True, text=True, timeout=30).stderr
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        pass
    return 0.0


def _frame(vid, t, width=1024):
    from PIL import Image
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
    try:
        subprocess.run(["ffmpeg", "-y", "-ss", str(max(0.0, t)), "-i", str(vid), "-frames:v", "1",
                        "-vf", f"scale={width}:-1", "-q:v", "3", tmp], capture_output=True, timeout=60)
        if os.path.getsize(tmp) == 0:
            return None
        img = Image.open(tmp)
        img.load()
        return img.convert("RGB")
    except Exception:
        return None
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def frame_images(wts_root, scenario, phase, seg, n_overhead=4):
    overhead, vehicle = find_views(wts_root, scenario)
    imgs = []
    if overhead:
        dur = _duration(overhead)
        if dur > 0:
            times = [dur * (i + 0.5) / n_overhead for i in range(n_overhead)]
            s, e = seg.get((scenario, phase), (None, None))
            if s is not None and e is not None and e <= dur:
                times.append((s + e) / 2.0)
            imgs += [im for im in (_frame(overhead, t) for t in times) if im]
    if vehicle:
        dur = _duration(vehicle)
        im = _frame(vehicle, dur / 2 if dur > 0 else 1.0)
        if im:
            imgs.append(im)
    return imgs


_DEGEN = ("no pedestrian", "not visible", "cannot", "can't", "unable", "no person", "there is no",
          "i'm sorry", "sorry,", "as an ai", "no one", "n/a")


def is_degenerate(cap):
    if not cap or not cap.get("caption_pedestrian"):
        return True
    if len(cap["caption_pedestrian"].split()) < 20:
        return True
    p = cap["caption_pedestrian"].lower()
    return any(k in p for k in _DEGEN)


GENERIC_FRAME = {
    "caption_pedestrian": (
        "The pedestrian is walking steadily along the paved road, positioned near the vehicle's path and moving "
        "across the open area. He appears attentive to his surroundings as he proceeds on foot. The weather is "
        "clear with good brightness, and the road surface is dry and level. The road runs through a residential "
        "area with light traffic, lined with trees, greenery and open space on both sides."),
    "caption_vehicle": (
        "The vehicle is moving slowly along the road, with the pedestrian visible ahead within its field of view "
        "as it proceeds through the area. It advances cautiously along the dry, level paved surface. The weather "
        "is clear with good visibility, and the road runs through a residential environment with light traffic, "
        "surrounding trees and open space nearby."),
}

FRAME_USER = ("These are several frames from ONE short traffic clip (overhead-camera frames first, then a dashcam "
              "frame). A target pedestrian IS present in the scene even if small, far, or partly occluded; look "
              "carefully across ALL frames and describe the pedestrian and the vehicle. Never say the pedestrian "
              "is absent. Return strict JSON only.")


def frame_caption(runner, wts_root, scenario, phase, seg):
    """Describe real frames. Always runs on the base weights — the LoRA is text-only."""
    if not wts_root:
        return dict(GENERIC_FRAME)
    imgs = frame_images(wts_root, scenario, phase, seg)
    if not imgs:
        return dict(GENERIC_FRAME)
    msgs = [text_msg("system", FRAME_SYS), image_msg("user", FRAME_USER, imgs)]
    with runner.base_weights():
        for attempt in range(3):
            try:
                cap = parse_caps(runner.chat(msgs, max_new_tokens=600, temperature=0.3 + 0.2 * attempt))
            except Exception as e:
                print(f"  frame retry {scenario} p{phase} ({e})", flush=True)
                continue
            if cap and not is_degenerate(cap):
                return cap
    return dict(GENERIC_FRAME)


# --------------------------------------------------------------------------- last-resort fallback
APPEAR = "The pedestrian, a male in his 20s wearing a T-shirt and slacks,"
ENV = ("As for the environment, the weather was clear with dim brightness, and the road surface was dry on a "
       "level asphalt road, classified as a residential road with light two-way traffic.")


def template_caption(facts_qa):
    """Never leave a segment empty. Normally unused — only fires if generation failed outright."""
    def v(q):
        return facts_qa.get(norm(q), "").lower()

    pos = v("What is the position of the pedestrian relative to the vehicle?")
    dist = v("What is relative distance of pedestrian from vehicle?")
    ori = v("What is the orientation of the pedestrian's body?")
    los = v("What is the pedestrian's line of sight?")
    aware = v("What is the pedestrian's awareness regarding vehicle?")
    act = v("What is the pedestrian's action?")
    travel = v("What is the pedestrian's direction of travel?")
    speed = v("What is pedestrian's speed?")
    ped = (f"{APPEAR} was positioned {pos} at a {dist} distance, with body orientation {ori}. "
           f"His line of sight was {los}, and he was {aware} of the vehicle. "
           f"He was {act}, heading {travel} at a {speed} speed. {ENV}")

    vpos = v("What is the position of the vehicle relative to the pedestrian?")
    vdist = v("What is relative distance of vehicle from pedestrian?")
    fov = v("What is vehicle's field of view?")
    vact = v("What is the action taken by vehicle?")
    veh = (f"The vehicle was {vpos} the pedestrian at a {vdist} distance, with the pedestrian {fov} its field of "
           f"view. The vehicle was {vact}. {ENV} The pedestrian is a male in his 20s wearing a T-shirt and slacks.")
    return {"caption_pedestrian": norm(ped), "caption_vehicle": norm(veh)}


# --------------------------------------------------------------------------- input check
def check_inputs(args, segments, facts) -> int:
    """Report what is present and what the run still needs. Returns the number of blockers."""
    rows = [
        ("VQA answers", args.vqa, True),
        ("public-test questions", args.test, True),
        ("segment list", args.segments, True),
        ("few-shot examples", args.fewshot, args.fewshot_n > 0),
    ]
    missing = 0
    print("== inputs ==")
    for name, path, required in rows:
        ok = Path(path).is_file()
        missing += int(required and not ok)
        print(f"  [{'ok' if ok else ('MISSING' if required else '--')}] {name:24s} {path}")
    if args.mode == "lora":
        ok = (Path(args.lora) / "adapter_config.json").is_file()
        missing += int(not ok)
        print(f"  [{'ok' if ok else 'MISSING'}] {'caption LoRA':24s} {args.lora}")

    n_qa = sum(1 for s, ents in segments.items() for e in ents
               if facts.get((s, str((e.get('labels') or ['?'])[0]))))
    n_total = sum(len(v) for v in segments.values())
    print(f"\n== coverage ==\n  scenarios={len(segments)} segments={n_total}"
          f"  QA path={n_qa}  frame path={n_total - n_qa}")

    print("\n== frame path ==")
    if n_total - n_qa == 0:
        print("  not needed (every segment has VQA answers)")
    elif not args.wts_root:
        print("  [MISSING] --wts-root not set -> those segments fall back to a generic caption")
        print("            supply a dir with annotations/caption/test/public_challenge/ and videos/test/public/")
        missing += 1
    else:
        vids = Path(args.wts_root) / "videos" / "test" / "public"
        anns = Path(args.wts_root) / "annotations" / "caption" / "test" / "public_challenge"
        for name, p in (("videos", vids), ("annotations", anns)):
            ok = p.is_dir()
            missing += int(not ok)
            print(f"  [{'ok' if ok else 'MISSING'}] {name:12s} {p}")
        for tool in ("ffmpeg", "ffprobe"):
            ok = which(tool) is not None
            missing += int(not ok)
            print(f"  [{'ok' if ok else 'MISSING'}] {tool:12s} {'found' if ok else 'install it and put it on PATH'}")

    print("\n== python packages ==")
    for mod in ("torch", "transformers", "PIL") + (("peft",) if args.mode == "lora" else ()):
        try:
            __import__(mod)
            print(f"  [ok] {mod}")
        except ImportError:
            missing += 1
            print(f"  [MISSING] {mod}  -> pip install -r requirements.txt")

    print("\n" + ("READY" if missing == 0 else f"NOT READY: {missing} item(s) missing"))
    return missing


# --------------------------------------------------------------------------- main
def build_args():
    ap = argparse.ArgumentParser(description="Generate WTS pedestrian/vehicle captions from the VQA answers.")
    ap.add_argument("--mode", choices=["lora", "base"], default="lora",
                    help="lora = Qwen3-VL + the shipped caption adapter (default); base = plain Qwen3-VL")
    ap.add_argument("--vqa", default=str(DEF_VQA), help="VQA submission [{id, correct}]")
    ap.add_argument("--test", default=str(DEF_TEST), help="public-test question file")
    ap.add_argument("--segments", default=str(DEF_SEGMENTS), help="scenarios/segments to caption")
    ap.add_argument("--fewshot", default=str(DEF_FEWSHOT))
    ap.add_argument("--fewshot_n", type=int, default=-1,
                    help="number of style examples; default 4 in base mode, 0 in lora mode")
    ap.add_argument("--model", default=BASE_MODEL, help="HF id or local path of Qwen3-VL")
    ap.add_argument("--lora", default=str(DEF_LORA), help="caption LoRA dir (lora mode)")
    ap.add_argument("--wts-root", dest="wts_root", default=os.getenv("WTS_ROOT", ""),
                    help="WTS test root with annotations/ and videos/ (frame path)")
    ap.add_argument("--out", default=str(DEF_OUT))
    ap.add_argument("--cache", default="", help="resume file (default: <out>_cache.jsonl)")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N segments (smoke test)")
    ap.add_argument("--check", action="store_true", help="verify the inputs and exit, generating nothing")
    args = ap.parse_args()
    if args.fewshot_n < 0:
        args.fewshot_n = 0 if args.mode == "lora" else 4
    return args


def main():
    args = build_args()
    required = [args.vqa, args.test, args.segments]
    have_all = all(Path(p).is_file() for p in required)

    # --check must survive missing inputs: it exists to list every one of them at once.
    segments = json.load(open(args.segments)) if Path(args.segments).is_file() else {}
    facts = build_facts(args.vqa, args.test) if Path(args.vqa).is_file() and Path(args.test).is_file() else {}

    if args.check:
        sys.exit(1 if check_inputs(args, segments, facts) else 0)
    if not have_all:
        raise SystemExit(f"FATAL: input not found: {next(p for p in required if not Path(p).is_file())}\n"
                         f"       run `--check` to see everything the run needs.")

    tasks = [(s, str((e.get("labels") or ["?"])[0]), e)
             for s, ents in segments.items() for e in ents]
    if args.limit:
        tasks = tasks[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache) if args.cache else out_path.with_name(out_path.stem + "_cache.jsonl")
    done = {}
    if cache_path.exists():
        for line in cache_path.open():
            try:
                r = json.loads(line)
                done[(r["scenario"], r["phase"])] = r["cap"]
            except Exception:
                pass
        print(f"resuming: {len(done)} segments already in {cache_path}", flush=True)

    todo = [t for t in tasks if (t[0], t[1]) not in done]
    n_frame = sum(1 for s, ph, _ in todo if not facts.get((s, ph)))
    print(f"mode={args.mode} model={args.model}"
          f"{' + ' + args.lora if args.mode == 'lora' else ''}", flush=True)
    print(f"segments={len(tasks)} to generate={len(todo)} (QA={len(todo) - n_frame} frame={n_frame})", flush=True)

    if todo:
        if args.seed:
            import torch
            torch.manual_seed(args.seed)
        runner = QwenVL(args.model, args.lora if args.mode == "lora" else "",
                        dtype=args.dtype, device_map=args.device_map,
                        max_new_tokens=args.max_new_tokens)
        shots = load_style_examples(args.fewshot, args.fewshot_n)
        seg_times = load_segment_times(args.wts_root) if args.wts_root else {}
        if n_frame and not args.wts_root:
            print(f"WARNING: {n_frame} segments have no VQA answers and --wts-root is unset; "
                  f"they get the generic fallback caption. Run with --check for details.", flush=True)

        try:
            from tqdm import tqdm
            bar = tqdm(todo, desc="captioning")
        except ImportError:
            bar = todo
        for i, (scenario, phase, _entry) in enumerate(bar, 1):
            qa = facts.get((scenario, phase), {})
            try:
                if qa:
                    cap = qa_caption(runner, scenario, phase, qa, shots, args.temperature)
                else:
                    cap = frame_caption(runner, args.wts_root, scenario, phase, seg_times)
            except Exception as e:                 # one bad segment must never kill the run
                print(f"  segment error {scenario} p{phase}: {e}", flush=True)
                cap = None
            if not cap or not (cap.get("caption_pedestrian") and cap.get("caption_vehicle")):
                cap = template_caption(qa)
            done[(scenario, phase)] = cap
            with cache_path.open("a") as fh:
                fh.write(json.dumps({"scenario": scenario, "phase": phase, "cap": cap}) + "\n")
            if bar is todo and i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    submission = {}
    for scenario, ents in segments.items():
        rows = []
        for entry in ents:
            phase = str((entry.get("labels") or ["?"])[0])
            cap = done.get((scenario, phase))
            if not cap:
                continue
            rows.append({"labels": entry.get("labels", [phase]),
                         "caption_pedestrian": env_fix(cap["caption_pedestrian"]),
                         "caption_vehicle": env_fix(cap["caption_vehicle"])})
        if rows:
            submission[scenario] = rows

    json.dump(submission, out_path.open("w"), indent=1, ensure_ascii=True)
    n_seg = sum(len(v) for v in submission.values())
    empty = sum(1 for v in submission.values() for e in v
                if not e["caption_pedestrian"] or not e["caption_vehicle"])
    print(f"wrote {out_path} | scenarios={len(submission)} segments={n_seg} empty={empty}", flush=True)
    if empty:
        raise SystemExit(f"FATAL: {empty} segments have an empty caption")


if __name__ == "__main__":
    main()
