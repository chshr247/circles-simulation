"""Caption and hashtags for one video, from whatever is racing in it.

No LLM and no pool of ten shuffled tags: the video is a race between named
things - countries, rappers, dogs - so those names ARE the topic and matching
them is a roster lookup. The winner's tag always goes in - its fans are the one
audience guaranteed to care - then two more of the faces on screen, then the
pack's generic filler so a video still carries five tags.
"""
import random

import packs

# Per pack: the filler tags when the roster runs out, and the text burned onto
# the frame. A hook has to work for the thing on screen - "PICK YOUR FLAG" is
# nonsense over a row of rappers - so the two live together.
GENERIC = {
    "countries": ["#simulation", "#marblerace", "#satisfying", "#countryballs",
                  "#flags", "#whowillwin", "#physics", "#asmr", "#fyp", "#viral"],
    "rappers":   ["#rap", "#hiphop", "#rapper", "#music", "#whowillwin",
                  "#marblerace", "#simulation", "#fyp", "#viral", "#battle"],
    "footballers": ["#football", "#soccer", "#futbol", "#goat", "#whowillwin",
                    "#marblerace", "#simulation", "#fyp", "#viral", "#legends"],
    "streamers": ["#streamer", "#twitch", "#youtuber", "#gaming", "#whowillwin",
                  "#marblerace", "#simulation", "#fyp", "#viral", "#clips"],
    "presidents": ["#politics", "#president", "#worldleaders", "#news",
                   "#whowillwin", "#marblerace", "#simulation", "#fyp",
                   "#viral", "#geopolitics"],
    "kpop":      ["#kpop", "#kpopfyp", "#idol", "#bias", "#whowillwin",
                  "#marblerace", "#simulation", "#fyp", "#viral", "#comeback"],
    # Deliberately bland: naming the industry is what gets the video pulled.
    "adult":     ["#guessher", "#whowillwin", "#marblerace", "#simulation",
                  "#satisfying", "#physics", "#fyp", "#viral", "#round1", "#top20"],
}
HOOKS = {
    "countries": ["WHO WILL WIN?", "LAST FLAG STANDING", "PICK YOUR FLAG",
                  "GUESS THE WINNER", "COMMENT YOUR COUNTRY", "BET ON ONE"],
    "rappers":   ["WHO WILL WIN?", "LAST RAPPER STANDING", "PICK YOUR GOAT",
                  "GUESS THE WINNER", "COMMENT YOUR FAVOURITE", "BET ON ONE"],
    "footballers": ["WHO WILL WIN?", "LAST ONE STANDING", "PICK YOUR GOAT",
                    "GUESS THE WINNER", "COMMENT YOUR PLAYER", "BET ON ONE"],
    "streamers": ["WHO WILL WIN?", "LAST STREAMER STANDING", "PICK YOUR MAIN",
                  "GUESS THE WINNER", "COMMENT YOUR FAVOURITE", "BET ON ONE"],
    "presidents": ["WHO WILL WIN?", "LAST LEADER STANDING", "PICK YOUR PRESIDENT",
                   "GUESS THE WINNER", "COMMENT YOUR COUNTRY", "BET ON ONE"],
    "kpop":      ["WHO WILL WIN?", "LAST GROUP STANDING", "PICK YOUR BIAS",
                  "GUESS THE WINNER", "COMMENT YOUR GROUP", "BET ON ONE"],
    "adult":     ["WHO WILL WIN?", "LAST ONE STANDING", "PICK ONE",
                  "GUESS THE WINNER", "COMMENT YOUR PICK", "BET ON ONE"],
}
FALLBACK = ["#whowillwin", "#marblerace", "#simulation", "#satisfying",
            "#physics", "#asmr", "#fyp", "#viral"]
MATCHED = 3      # three named contenders; five would read as a tag wall


def hooks(pack):
    return HOOKS.get(pack, HOOKS["countries"])


def flag(cc):
    """The regional-indicator pair for a country code - no emoji table needed."""
    return "".join(chr(0x1F1E6 + ord(c) - ord("a")) for c in cc.lower())


def pick(pack, ids, winner=None, n=5):
    """Up to `n` hashtags for one video, the things in it first."""
    names = packs.roster(pack)
    out = []
    if winner in names:
        out.append("#" + packs.slug(names[winner]))
    rest = [c for c in ids if c != winner and c in names]
    random.shuffle(rest)
    out += ["#" + packs.slug(names[c]) for c in rest[:MATCHED - len(out)]]
    filler = GENERIC.get(pack, FALLBACK)[:]
    random.shuffle(filler)
    for t in filler:
        if len(out) >= n:
            break
        if t not in out:
            out.append(t)
    return out[:n]


def caption(hook, pack, ids, winner=None):
    """Hook, tags, then the line-up - and that ORDER is load-bearing.

    The uploader counts text_extra offsets in Python code points; TikTok reads
    them as UTF-16 units. A flag emoji is 2 against 4, so one of them in front
    of a hashtag shifts every offset behind it. Measured 2026-08-20 with the
    flags on the first line: #southafrica went out as 48-60, which lands in the
    middle of a surrogate pair, and TikTok answered status_code 5 "Invalid
    parameters" - after the whole 55 MB had already gone up. So everything
    ahead of the last tag stays inside the BMP, and the flags go behind it
    where no offset reaches.

    The winner is never named in words: the video is the answer to the hook.
    """
    names = packs.roster(pack)
    if pack == "countries":
        line = " ".join(flag(c) for c in ids)
    else:
        line = " vs ".join(packs.display(names[c]) for c in ids if c in names)
    return "%s\n%s\n%s" % (hook, " ".join(pick(pack, ids, winner)), line)


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):   # a console with no flags in it
        sys.stdout.reconfigure(errors="replace")
    country_tags = {"#" + packs.slug(v) for v in packs.roster("countries").values()}
    t = pick("countries", ["br", "ar", "us", "de"], "br")
    assert t[0] == "#brazil", t
    assert len(t) == 5 == len(set(t)), t
    assert len([x for x in t if x in country_tags]) == MATCHED, t

    # an id off the roster costs its tag and nothing else
    blind = pick("countries", ["zz", "qq"], "zz")
    assert len(blind) == 5 and not (set(blind) & country_tags), blind

    # two uploads of the same matchup must not carry identical text
    same = {" ".join(pick("countries", ["br", "ar", "us", "de", "fr"], "br"))
            for _ in range(30)}
    assert len(same) > 3, "tags are not rotating"

    c = caption(hooks("countries")[0], "countries", ["br", "ar"], "br")
    assert flag("br") in c and "#brazil" in c and len(c) < 2200, c
    # The invariant the order in caption() exists for: nothing outside the BMP
    # may sit in front of the last hashtag, or its offset goes out wrong by
    # however many astral characters preceded it.
    assert max(map(ord, c[:c.rindex("#")])) < 0x10000, c
    print(c)

    # every pack captions, and none of them smuggles an astral character in
    for p in packs.all_packs():
        ids = list(packs.roster(p))[:4]
        cp = caption(hooks(p)[0], p, ids, ids[0])
        assert len(pick(p, ids, ids[0])) == 5, (p, cp)
        assert max(map(ord, cp[:cp.rindex("#")])) < 0x10000, cp
        assert len(cp) < 2200 and "#" + packs.slug(packs.roster(p)[ids[0]]) in cp, cp
        print("\n--- %s ---\n%s" % (p, cp))
