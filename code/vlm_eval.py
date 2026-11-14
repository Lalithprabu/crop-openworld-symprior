"""Local vision-language chat models (Ollama) on the same benchmark as SymPrior.

Tasks (every photograph, options shuffled per photograph with a fixed seed, temperature 0):
  T1  names        choose the field's class from its class names                      (= zero-shot, closed set)
  T2  names+desc   the same, with the two symptom descriptions of each class            (= the information SymPrior gets)
  T3  naming       unknown disease: rank the top 3 from the naming bank minus the known classes
                   (field B: yellow vein mosaic photos, 12 options; field C: each photo, its class held out, 9 options)
  T4  1-shot       one labelled photo per class shown in the prompt (first 3 k=1 episodes of episodes.json) + descriptions
Results: ../results_vlm/<model>.jsonl, one line per call; reruns skip finished calls, so the run can be stopped and restarted.

Usage: python vlm_eval.py [model ...]        (default: chosen from the GPU memory; models are pulled if missing)
"""
import os, sys, csv, json, time, base64, io, random, subprocess, urllib.request, urllib.error
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
EXT = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "external")          # A:\PhD\Novelty\external
IMG_ROOT = {"A": os.path.join(EXT, "ext_a"), "B": os.path.join(EXT, "ext_b"), "C": os.path.join(ROOT, "data_tomato", "images")}
BENCH = os.path.join(ROOT, "release", "benchmark"); OUT = os.path.join(ROOT, "results_vlm"); os.makedirs(OUT, exist_ok=True)
URL = "http://localhost:11434/api/chat"
DESC = json.load(open(os.path.join(ROOT, "release", "descriptions", "descriptions.json"), encoding="utf-8"))
BANK = {"A": DESC["okra_bank_15"], "B": DESC["okra_bank_15"], "C": DESC["tomato_bank_16"]}
CROP = {"A": "okra", "B": "okra", "C": "tomato"}
ART = {"okra": "an okra", "tomato": "a tomato"}
SYSTEM = "You are an expert plant pathologist. Answer only with the JSON requested."


def load_index(ds):
    rows = list(csv.DictReader(open(os.path.join(BENCH, f"index_{ds}.csv"), encoding="utf-8")))
    for r in rows:
        r["path"] = os.path.join(IMG_ROOT[ds], r["file"] if ds != "C" else r["img"]); r["label"] = int(r["label"])
    classes = [None] * (max(r["label"] for r in rows) + 1)
    for r in rows: classes[r["label"]] = r["class"]
    return rows, classes


_cache = {}
def b64(path, side=640):
    if (path, side) not in _cache:
        im = Image.open(path).convert("RGB"); im.thumbnail((side, side))
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=90); v = base64.b64encode(buf.getvalue()).decode()
        if len(_cache) > 3000: _cache.clear()
        _cache[(path, side)] = v
    return _cache[(path, side)]


def desc_lines(ds, names):
    return "\n".join(f"- {n}: " + "; ".join(BANK[ds][n]) for n in names)


THINK = {}
def chat(model, prompt, images, schema, ctx=4096):
    body = {"model": model, "stream": False, "format": schema, "keep_alive": "30m",
            "options": {"temperature": 0, "seed": 0, "num_ctx": ctx},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt, "images": images}]}
    if THINK.get(model, True): body["think"] = False
    for attempt in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            return r["message"]["content"]
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")
            if "think" in msg.lower() and "think" in body:
                THINK[model] = False; body.pop("think"); continue
            err = f"HTTP {e.code}: {msg[:200]}"
        except Exception as e:
            err = repr(e)[:200]
        time.sleep(3)
    return "ERROR " + err


def enum1(opts): return {"type": "object", "properties": {"answer": {"type": "string", "enum": opts}}, "required": ["answer"]}
def enum3(opts): return {"type": "object", "properties": {k: {"type": "string", "enum": opts} for k in ("first", "second", "third")},
                         "required": ["first", "second", "third"]}


def jobs_for(ds, task, rows, classes):
    """yields (key, prompt, image paths, schema, truth, options, ctx)"""
    crop = CROP[ds]
    if task in ("T1", "T2"):
        for r in rows:
            opts = classes[:]; random.Random(f"{task}{ds}{r['idx'] if 'idx' in r else r['img']}").shuffle(opts)
            p = f"This photograph shows {ART[crop]} leaf. Which one of these is it?\nOptions: " + ", ".join(opts)
            if task == "T2": p += "\nWhat each option looks like:\n" + desc_lines(ds, opts)
            p += '\nReply as {"answer": "<one option>"}.'
            yield (f"{task}|{ds}|{r.get('idx', r.get('img'))}", p, [r["path"]], enum1(opts), r["class"], opts, 4096)
    elif task == "T3":
        bank = list(BANK[ds])
        for r in rows:
            if ds == "B" and r["class"] != "Yellow vein mosaic": continue
            known = [c for c in classes if c != r["class"]]
            opts = [n for n in bank if n not in known]; random.Random(f"T3{ds}{r.get('idx', r.get('img'))}").shuffle(opts)
            p = (f"This photograph shows {ART[crop]} leaf with a disorder. Which disorder is it? "
                 f"Give the three most likely options, most likely first.\nOptions:\n" + desc_lines(ds, opts) +
                 '\nReply as {"first": "<option>", "second": "<option>", "third": "<option>"}.')
            yield (f"T3|{ds}|{r.get('idx', r.get('img'))}", p, [r["path"]], enum3(opts), r["class"], opts, 4096)
    elif task == "T4":
        E = json.load(open(os.path.join(BENCH, "episodes.json")))[f"fewshot_{ds}"]["k1"]
        for e, sup in enumerate(E[:3]):
            sup = sorted(sup, key=lambda i: rows[i]["label"]); sg = {rows[i]["group"] for i in sup}
            q = [i for i in range(len(rows)) if rows[i]["group"] not in sg]
            if ds == "C": q = sorted(random.Random(f"T4C{e}").sample(q, 150))
            names = [rows[i]["class"] for i in sup]
            ref = "\n".join(f"Image {j + 1}: {n} (" + "; ".join(BANK[ds][n]) + ")" for j, n in enumerate(names))
            for i in q:
                opts = names[:]; random.Random(f"T4{ds}{e}{i}").shuffle(opts)
                p = (f"The first {len(sup)} images are labelled reference photographs of {crop} leaves, one per class:\n{ref}\n"
                     f"The last image is a new photograph. Which class is it?\nOptions: " + ", ".join(opts) +
                     '\nReply as {"answer": "<one option>"}.')
                yield (f"T4|{ds}|e{e}|{i}", p, [rows[j]["path"] for j in sup] + [rows[i]["path"]], enum1(opts), rows[i]["class"], opts, 16384)


def t4_subset(ds, jobs):
    """the fixed class-stratified T4 query subset (field B 12, tomato 4 per class and episode), independent of what is done"""
    cap, keep = {"B": 12, "C": 4}[ds], set()
    for e in range(3):
        js = [j for j in jobs if j[0].split("|")[2] == f"e{e}"]; random.Random(f"q{ds}{e}").shuffle(js)
        for c in sorted({j[4] for j in js}):
            keep |= set([j[0] for j in js if j[4] == c][:cap])
    return keep


def gpu_mem_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout.strip().splitlines()[0]
        name, mem = out.rsplit(",", 1); print("GPU:", name.strip(), mem.strip(), "MB"); return int(mem) / 1024
    except Exception:
        print("GPU: none found (CPU only - slow)"); return 0


def ensure(model):
    have = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    if model.split(":")[0] in have and (":" not in model or model in have): return True
    print(f"pulling {model} ..."); return subprocess.run(["ollama", "pull", model]).returncode == 0


class Tee:
    def __init__(self, *f): self.f = f
    def write(self, x):
        for f in self.f: f.write(x); f.flush()
    def flush(self):
        for f in self.f: f.flush()


def main():
    sys.stdout = Tee(sys.__stdout__, open(os.path.join(ROOT, "vlm_log.txt"), "a", encoding="utf-8"))
    print(subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip())
    mem = gpu_mem_gb()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; quick = "--t4quick" in sys.argv
    models = args or (["qwen3-vl:8b", "gemma3:12b"] if mem >= 11 else ["qwen3-vl:8b", "gemma3:4b"] if mem >= 6
                              else ["qwen3-vl:4b", "gemma3:4b"])
    ok = []
    for m in models:
        if ensure(m): ok.append(m)
        elif m.startswith("qwen3-vl") and ensure("qwen2.5vl:7b"): ok.append("qwen2.5vl:7b")
    print("models:", ok)
    data = {ds: load_index(ds) for ds in "ABC"}
    for ds, (rows, cl) in data.items():
        miss = [r["path"] for r in rows if not os.path.exists(r["path"])]
        print(f"field {ds}: {len(rows)} photos, classes {cl}, missing files {len(miss)}", miss[:2])
    blocks = [("B", "T1"), ("B", "T2"), ("B", "T3"), ("C", "T1"), ("C", "T2"), ("C", "T3"), ("B", "T4"), ("C", "T4"), ("A", "T1"), ("A", "T2")]
    runs = [(ds, t, m) for ds, t in blocks for m in ok]
    if quick:       # --t4quick: one labelled photo per class in the prompt, a class-stratified subset of the queries, smaller images,
                    # the faster model first; then the rest of the original plan
        order = sorted(ok, key=lambda m: "gemma" not in m)
        runs = [(ds, "T4", m) for m in order for ds in ("B", "C")] + [r for r in runs if r[1] != "T4"]
    for ds, task, m in runs:
        if True:
            fn = os.path.join(OUT, m.replace(":", "_").replace("/", "_") + ".jsonl")
            done = set()
            if os.path.exists(fn):
                for line in open(fn, encoding="utf-8"):
                    d = json.loads(line)
                    if not d["raw"].startswith("ERROR"): done.add(d["key"])
            alljobs = list(jobs_for(ds, task, *data[ds])); side = 640
            if quick and task == "T4":
                side, keys = 448, t4_subset(ds, alljobs); alljobs = [j for j in alljobs if j[0] in keys]
            todo = [j for j in alljobs if j[0] not in done]
            print(f"\n[{time.strftime('%H:%M')}] {m} field {ds} {task}: {len(todo)} calls to do", flush=True)
            t0, n, right = time.time(), 0, 0
            with open(fn, "a", encoding="utf-8") as f:
                for key, p, imgs, schema, truth, opts, ctx in todo:
                    t = time.time(); raw = chat(m, p, [b64(x, side) for x in imgs], schema, ctx)
                    try: ans = json.loads(raw); pred = ans.get("answer", ans.get("first"))
                    except Exception: ans, pred = None, None
                    f.write(json.dumps({"key": key, "model": m, "truth": truth, "pred": pred, "ans": ans, "options": opts,
                                        "raw": raw[:500], "sec": round(time.time() - t, 2), "side": side}) + "\n"); f.flush()
                    n += 1; right += pred == truth
                    if n % 25 == 0 or n == len(todo):
                        el = time.time() - t0
                        print(f"  {n}/{len(todo)}  top-1 so far {100 * right / n:.1f}%  {el / n:.1f} s/call  "
                              f"~{(len(todo) - n) * el / n / 60:.0f} min left in this block", flush=True)
    print("\nAll blocks finished.")


if __name__ == "__main__":
    main()
