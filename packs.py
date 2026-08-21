"""Categories. One pack is one thing the balls can be: countries, rappers, dogs.

A pack is a text file - `packs/<pack>.txt`, one `id Display Name` per line -
and a folder of pictures next to it that fills itself on first use. The rest of
the pipeline only ever sees ids, so a new category is a text file and no code.

The display name is also the lookup: flags go to flagcdn by their two-letter
code, everything else asks Wikipedia for the page's thumbnail. A parenthesised
disambiguator ("Future (rapper)") is what Wikipedia needs and not what the
caption should say, so it is stripped everywhere but the request.
"""
import json, os, random, re, time, unicodedata, urllib.parse, urllib.request

from PIL import Image, ImageStat

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "packs")
FLAG_URL = "https://flagcdn.com/w320/{}.png"
WIKI_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
UA = {"User-Agent": "circles/1 (+https://github.com/; contact: repo owner)"}
DEFAULT = "countries"


def all_packs():
    return sorted(f[:-4] for f in os.listdir(DIR) if f.endswith(".txt"))


def roster(pack):
    """{id: display name}, in file order - dicts keep it since 3.7."""
    out = {}
    with open(os.path.join(DIR, pack + ".txt"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                i, name = line.split(None, 1)
                out[i] = name
    return out


def display(name):
    """What a human reads: the disambiguator is Wikipedia's, not the caption's."""
    return name.split(" (")[0]


def slug(name):
    """A hashtag body: ASCII letters and digits, nothing else.

    Accents are folded rather than dropped - #kylianmbapp is not a hashtag
    anybody searches for.
    """
    flat = unicodedata.normalize("NFKD", display(name))
    return re.sub(r"[^a-z0-9]", "", flat.encode("ascii", "ignore").decode().lower())


def _get(url, timeout=20):
    """One retry, because Wikimedia throttles a roster fetched back to back and
    answers 429 - and a pack that loses a third of its faces to a burst limit
    is a worse video, not an error anybody sees."""
    for attempt in (0, 1):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if attempt:
                raise
            time.sleep(2)


def _wiki_urls(name):
    """The Wikipedia page's lead image, best first, empty if it has none.

    Both sizes are offered because the small one is sometimes a black frame -
    a video still the API thumbnailed at the wrong second - while the full
    picture behind it is fine. The thumbnail is taken at whatever width the
    API hands back: Wikimedia serves only the widths it lists and answers 400
    to anything else, so asking for a rounder number costs the picture.
    """
    url = WIKI_URL.format(urllib.parse.quote(name.replace(" ", "_"), safe=""))
    page = json.loads(_get(url))
    return [src for key in ("thumbnail", "originalimage")
            for src in [(page.get(key) or {}).get("source")] if src]


MIN_LIGHT = 12.0    # a picture darker than this on average is a black rectangle


def usable(path):
    """Wikipedia hands back the odd all-black frame - a video still, or a
    picture that was replaced by nothing. It renders as a hole where a face
    should be, so it is not a picture as far as a pack is concerned."""
    return ImageStat.Stat(Image.open(path).convert("L")).mean[0] >= MIN_LIGHT


def image_path(pack, item, fetch=True):
    """The cached picture for one item, downloading it once. None if there is
    no picture to be had - a Wikipedia page without one is a roster line that
    quietly stops being drawn, not a dead render at frame zero."""
    path = os.path.join(DIR, pack, item + ".png")
    if os.path.exists(path):
        return path
    if not fetch:
        return None
    names = roster(pack)
    if item not in names:
        raise SystemExit("%s is not in pack %s" % (item, pack))
    urls = ([FLAG_URL.format(item)] if pack == "countries"
            else _wiki_urls(names[item]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in urls:
        with open(path, "wb") as f:          # .png is the cache name, not a claim
            f.write(_get(url, timeout=30))   # about the bytes - PIL sniffs those
        if usable(path):
            return path
    if os.path.exists(path):                 # only a black frame came back
        os.remove(path)
    return None


def pick(pack, spec):
    """The ids for one video: `randomN`, or an explicit comma-separated list.

    Every pick is downloaded here rather than at frame zero, so a roster line
    whose picture has gone missing costs that line and not the run.
    """
    names = roster(pack)
    if not spec.startswith("random"):
        want = [c.strip().lower() for c in spec.split(",") if c.strip()]
        missing = [c for c in want if c not in names]
        if missing:
            raise SystemExit("not in pack %s: %s" % (pack, ", ".join(missing)))
        return want
    n = int(spec[6:] or 8)
    order = list(names)
    random.shuffle(order)
    out = []
    for item in order:
        if len(out) >= n:
            break
        try:
            path = image_path(pack, item)
            if path and usable(path):
                out.append(item)
                continue
        except Exception:
            pass
        print("  no usable picture for %s, skipping" % item)
    if len(out) < n:
        raise SystemExit("pack %s only yielded %d of %d pictures" % (pack, len(out), n))
    return out


if __name__ == "__main__":
    assert slug("Kylian Mbappe") == "kylianmbappe"
    assert slug("Future (rapper)") == "future"
    assert slug("Tyler, the Creator") == "tylerthecreator"
    assert slug("50 Cent") == "50cent"
    assert display("Ninja (gamer)") == "Ninja"
    every = all_packs()
    assert DEFAULT in every and len(every) > 3, every
    for p in every:
        r = roster(p)
        assert len(r) >= 8, (p, len(r))          # random8 must have somebody to draw
        assert len(set(map(slug, r.values()))) == len(r), "duplicate hashtag in " + p
    # countries ship with their pictures committed, so this needs no network
    assert image_path("countries", "br", fetch=False), "the flags moved"
    assert usable(image_path("countries", "br", fetch=False))
    assert pick("countries", "br,ar") == ["br", "ar"]
    print("packs ok: %s" % ", ".join("%s(%d)" % (p, len(roster(p))) for p in every))
