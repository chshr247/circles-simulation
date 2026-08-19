"""Caption and hashtags for one video, from the flags that are in it.

No LLM and no pool of ten shuffled tags: the video is a race between named
countries, so the countries ARE the topic and matching them is a dict lookup.
The winner's tag always goes in - that country's feed is the one audience
guaranteed to care - then two more of the flags on screen, then generic filler
so a video still carries five tags when everything else is unknown.
"""
import random

# The 44 codes in main.POOL plus the two extra flags on disk. A code missing
# here costs its tag, never a crash: pick() skips what it cannot name.
NAMES = {
    "ae": "uae", "ar": "argentina", "au": "australia", "be": "belgium",
    "br": "brazil", "ca": "canada", "ch": "switzerland", "cl": "chile",
    "cn": "china", "co": "colombia", "cz": "czechia", "de": "germany",
    "dk": "denmark", "eg": "egypt", "es": "spain", "fi": "finland",
    "fr": "france", "gb": "uk", "gr": "greece", "id": "indonesia",
    "ie": "ireland", "il": "israel", "in": "india", "it": "italy",
    "jp": "japan", "kr": "korea", "ma": "morocco", "mx": "mexico",
    "ng": "nigeria", "nl": "netherlands", "no": "norway", "nz": "newzealand",
    "pe": "peru", "ph": "philippines", "pl": "poland", "pt": "portugal",
    "ro": "romania", "ru": "russia", "sa": "saudiarabia", "se": "sweden",
    "th": "thailand", "tr": "turkey", "ua": "ukraine", "us": "usa",
    "vn": "vietnam", "za": "southafrica",
}

GENERIC = ["#simulation", "#marblerace", "#satisfying", "#countryballs",
           "#flags", "#whowillwin", "#physics", "#asmr", "#fyp", "#viral"]

# What gets burned onto the frame AND opens the caption, so it is one list.
HOOKS = ["WHO WILL WIN?", "LAST FLAG STANDING", "PICK YOUR FLAG",
         "GUESS THE WINNER", "COMMENT YOUR COUNTRY", "BET ON ONE"]

COUNTRY_TAGS = {"#" + v for v in NAMES.values()}
MATCHED = 3      # three named countries; five would read as a tag wall


def flag(cc):
    """The regional-indicator pair for a country code - no emoji table needed."""
    return "".join(chr(0x1F1E6 + ord(c) - ord("a")) for c in cc.lower())


def pick(ccs, winner=None, n=5):
    """Up to `n` hashtags for one video, the countries in it first."""
    out = []
    if winner in NAMES:
        out.append("#" + NAMES[winner])
    rest = [c for c in ccs if c != winner and c in NAMES]
    random.shuffle(rest)
    out += ["#" + NAMES[c] for c in rest[:MATCHED - len(out)]]
    filler = GENERIC[:]
    random.shuffle(filler)
    for t in filler:
        if len(out) >= n:
            break
        if t not in out:
            out.append(t)
    return out[:n]


def caption(hook, ccs, winner=None):
    """Hook, the flags that raced, tags. The winner is never named in words -
    the whole video is the answer to the hook and the caption shows first."""
    return "%s %s\n%s" % (hook, " ".join(flag(c) for c in ccs),
                          " ".join(pick(ccs, winner)))


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):   # a console with no flags in it
        sys.stdout.reconfigure(errors="replace")
    t = pick(["br", "ar", "us", "de"], "br")
    assert t[0] == "#brazil", t
    assert len(t) == 5 == len(set(t)), t
    assert len([x for x in t if x in COUNTRY_TAGS]) == MATCHED, t

    # an unknown code costs its tag and nothing else
    blind = pick(["zz", "qq"], "zz")
    assert len(blind) == 5 and not (set(blind) & COUNTRY_TAGS), blind

    # two uploads of the same matchup must not carry identical text
    same = {" ".join(pick(["br", "ar", "us", "de", "fr"], "br")) for _ in range(30)}
    assert len(same) > 3, "tags are not rotating"

    c = caption(HOOKS[0], ["br", "ar"], "br")
    assert flag("br") in c and "#brazil" in c and len(c) < 2200, c
    print(c)
