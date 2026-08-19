"""Post what CI rendered to TikTok, from the desk that holds the cookie.

CI makes the video and throws its disk away; the uploader is a browser session
that cannot live on a runner. So this is the other half of the loop: it pulls
the artifacts this repo's workflow left, then hands one video to the patched
TiktokAutoUploader fork with the caption that was written beside it.

    python post.py                # fetch, then post one
    python post.py -n 2           # ...or two
    python post.py --private      # post unlisted - for the first run
    python post.py --fetch-only

Needs the `gh` CLI authenticated, and a .env beside this file (see
.env.example) pointing at the fork. Run it from Task Scheduler as often as you
like: it posts `-n` videos and leaves the rest queued in out/.
"""
import argparse, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
FETCHED = os.path.join(OUT, ".fetched")   # artifact names already downloaded
POSTED = os.path.join(OUT, ".posted")     # mp4 names already on the account
PREFIX = "video-"

# Printed by the fork on a real upload and only there - upstream exits 0 on
# paths that posted nothing, so this line is the difference between "sent" and
# "silently didn't". If it is missing, the checkout is unpatched.
CREATION_ID = re.compile(r"^creation_id=(\S+)", re.M)


def env(name, default=""):
    """Environment first, then .env. The same file the fork's runbook uses."""
    if name in os.environ:
        return os.environ[name]
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(name + "="):
                    return line[len(name) + 1:].strip().strip('"')
    return default


def seen(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(f.read().split())


def mark(path, name):
    with open(path, "a", encoding="utf-8") as f:
        f.write(name + "\n")


def gh(*args):
    r = subprocess.run(["gh", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("gh %s failed:\n%s" % (" ".join(args), (r.stderr or "").strip()))
    return r.stdout


def fetch():
    """Download every artifact this machine has not taken yet."""
    taken = seen(FETCHED)
    raw = gh("api", "--paginate", "repos/{owner}/{repo}/actions/artifacts?per_page=100")
    # --paginate concatenates one JSON object per page rather than merging them.
    arts = [a for page in raw.replace("}{", "}\n{").splitlines() if page.strip()
            for a in json.loads(page).get("artifacts", [])]
    # Oldest first: the queue below goes by mtime, and a batch downloaded
    # backwards would post the newest video first.
    todo = sorted((a for a in arts if a["name"].startswith(PREFIX)
                   and not a["expired"] and a["name"] not in taken),
                  key=lambda a: a["created_at"])
    if not todo:
        print("nothing new to fetch")
        return
    os.makedirs(OUT, exist_ok=True)
    for a in todo:
        print("downloading %s (%.0f MB)" % (a["name"], a["size_in_bytes"] / 1e6))
        try:
            gh("run", "download", a["name"][len(PREFIX):],
               "--name", a["name"], "--dir", OUT)
        except SystemExit as e:
            # One bad artifact must not stall every one behind it forever.
            print("  skipped: %s" % e)
            continue
        mark(FETCHED, a["name"])   # only after it landed, so a cut download retries


def tau_python(root):
    exe = env("TIKTOK_TAU_PYTHON")
    if exe:
        return exe
    for rel in ("Scripts/python.exe", "bin/python"):
        p = os.path.join(root, ".venv", rel)
        if os.path.exists(p):
            return p
    sys.exit("no venv under %s - set TIKTOK_TAU_PYTHON"
             % os.path.join(root, ".venv"))


def upload(mp4, caption, private):
    root, user = env("TIKTOK_TAU_DIR"), env("TIKTOK_TAU_USER")
    if not root or not user:
        sys.exit("set TIKTOK_TAU_DIR and TIKTOK_TAU_USER in .env - see .env.example")
    proxy, ua = env("TIKTOK_PROXY"), env("TIKTOK_UA")
    if not proxy:
        print("WARNING: TIKTOK_PROXY unset - this posts from the real IP")
    cmd = [tau_python(root), "cli.py", "upload", "-u", user,
           "-v", os.path.abspath(mp4), "-t", caption,
           "-vi", "1" if private else "0"]
    if proxy:
        cmd += ["-p", proxy]

    e = dict(os.environ)
    # The flag reaches the uploader's requests session; the variable is what the
    # patched login browser and the node signer read. Miss the second and the
    # signature is computed over the real IP, which is the point of both.
    if proxy:
        e["TIKTOK_PROXY"] = proxy
    if ua:
        e["TIKTOK_UA"] = ua
    if env("TIKTOK_TAU_BROWSERS"):
        e["PLAYWRIGHT_BROWSERS_PATH"] = env("TIKTOK_TAU_BROWSERS")
    # Without this a piped python writes cp1252 on Windows, and the one thing
    # worth reading - the failure, which quotes the caption - arrives mangled.
    e["PYTHONIOENCODING"] = "utf-8"

    # cwd is load-bearing: the fork reads ./config.txt and resolves CookiesDir
    # against the working directory.
    r = subprocess.run(cmd, cwd=root, env=e, capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       timeout=int(env("TIKTOK_TAU_TIMEOUT", "1800")))
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if r.returncode != 0:
        sys.exit("upload failed (exit %d):\n%s" % (r.returncode, out[-1500:]))
    m = CREATION_ID.search(out)
    if not m:
        # No line, no post. Marking it sent here would lose the video for good.
        sys.exit("exited 0 but printed no creation_id - either the checkout is "
                 "unpatched (apply tau-synergy.patch) or the upload failed "
                 "quietly:\n%s" % out[-1500:])
    return m.group(1)


def queue():
    """Fetched videos that still have a caption and have not gone out."""
    done = seen(POSTED)
    if not os.path.isdir(OUT):
        return []
    return sorted((os.path.join(OUT, f) for f in os.listdir(OUT)
                   if f.endswith(".mp4") and f not in done
                   and os.path.exists(os.path.join(OUT, f[:-4] + ".meta.json"))),
                  key=os.path.getmtime)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1, help="how many to post this run")
    ap.add_argument("--private", action="store_true", help="post unlisted")
    ap.add_argument("--fetch-only", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):     # captions are full of flags
        sys.stdout.reconfigure(errors="replace")
    # `gh api repos/{owner}/{repo}` resolves the placeholders from the working
    # directory's remote, so this cannot run from wherever a scheduler happened
    # to start it. One chdir here beats a WorkingDirectory nobody set.
    os.chdir(HERE)

    if not a.no_fetch:
        fetch()
    if a.fetch_only:
        return
    q = queue()[:a.n]
    print("queue: %d video(s) to post now" % len(q))
    for i, mp4 in enumerate(q):
        with open(mp4[:-4] + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        print("posting %s%s\n%s" % (os.path.basename(mp4),
                                    " (private)" if a.private else "",
                                    meta["caption"]))
        cid = upload(mp4, meta["caption"], a.private)
        mark(POSTED, os.path.basename(mp4))
        # 60 MB a video, three a day: kept, this fills a 50 GB server disk
        # inside a year and nothing else here would ever notice. The meta file
        # stays as the record, and CI holds the mp4 for five more days.
        os.remove(mp4)
        print("posted: %s" % cid)
        if i + 1 < len(q):
            time.sleep(60)     # two uploads back to back look like a bot


if __name__ == "__main__":
    main()
