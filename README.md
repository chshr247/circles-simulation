# circles-simulation

Balls on tethers in a 9:16 arena. They bounce, cut each other's tethers,
speed up on every ricochet; the last one alive wins. One file renders it,
`main.py`, straight to mp4 through ffmpeg with a melody synthesised from the
note order of a real track.

```bash
python main.py --pack random --hook random           # one video into out/
python main.py --pack rappers --countries random8    # one category
python main.py --countries br,ar,us --winner ar      # a rigged matchup
python main.py --finalists ua,us --winner ua         # ...and a rigged final two
python main.py --preview 3                           # first 3 seconds only
python main.py --selftest
```

## Packs

What races is a category, not always countries: `countries`, `rappers`,
`footballers`, `adult`, `streamers`, `presidents`, `kpop` - sixteen of each, the
ones people argue about. A pack is `packs/<pack>.txt`, one `id Display Name`
per line, and a folder beside it that fills itself the
first time the pack is drawn - flags come from flagcdn by their code,
everything else takes the lead picture off the English Wikipedia page the
display name names. So a new category is a text file and no code.

Only the flags are committed. Every other picture is fetched and cached under
`packs/<pack>/`, gitignored: a roster line whose page has lost its picture is
skipped at pick time and costs that line, not the run.

A parenthesised disambiguator is Wikipedia's and not the caption's -
`future Future (rapper)` asks for the right page and still says "Future" on
the video.

```bash
python packs.py    # self-check: every roster parses, no two ids share a hashtag
```

`sim.html` is the tuning prototype and reads the same rosters, so it needs to
be served rather than opened off disk - `fetch` of a local file is blocked and
only the pack picker breaks, but the picker is the point:

```bash
python -m http.server 8777   # then http://localhost:8777/sim.html
```

## The publishing loop

Two halves, because they cannot live in the same place.

**CI renders.** `.github/workflows/render.yml` runs three times a day, makes one
video, and leaves it as an artifact `video-<run id>` together with a
`.meta.json` carrying the caption. Nothing is posted from there: the uploader is
a logged-in web session, and a runner is the wrong machine to keep a TikTok
cookie on.

**The desk posts.** `post.py` pulls the artifacts this machine has not taken and
hands one to the patched [TiktokAutoUploader][tau] fork with the caption that
was written beside it.

```bash
python post.py --private     # first run: fetch, post one unlisted, check it
python post.py               # from then on
```

Setup, once:

1. `cp .env.example .env` and fill in `TIKTOK_TAU_DIR` and `TIKTOK_TAU_USER`.
2. Give the fork a cookie saved under exactly that name. Its own account: two
   channels behind one profile get flagged together.
3. `gh auth login`, if this machine has not already.
4. A scheduler - systemd timer or Task Scheduler - calling `python post.py` on
   the same slots CI renders on. It posts one video per run unconditionally, so
   the timer IS the rate: three slots a day for three renders a day.

The fork must carry the patch that prints `creation_id=` on a successful
upload — upstream exits 0 on paths that posted nothing, and `post.py` refuses to
mark a video sent without that line rather than lose it.

## Captions

`tags.py`, no LLM call. Whatever raced in the video is the topic, so the caption
is the hook, the line-up, and five hashtags: the winner first, two more of the
faces on screen, then filler from the pack's own list - a rapper video is not
tagged #countryballs. Which of the filler and in what order is random, so two
videos of the same matchup do not carry identical text.

```bash
python tags.py     # self-check, also what CI runs before rendering
```

[tau]: https://github.com/makiisthenes/TiktokAutoUploader
