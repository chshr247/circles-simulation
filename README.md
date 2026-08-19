# circles-simulation

Flag balls on tethers in a 9:16 arena. They bounce, cut each other's tethers,
speed up on every ricochet; the last one alive wins. One file renders it,
`main.py`, straight to mp4 through ffmpeg with a melody synthesised from the
note order of a real track.

```bash
python main.py --countries random8 --hook random     # one video into out/
python main.py --countries br,ar,us --winner ar      # a rigged matchup
python main.py --preview 3                           # first 3 seconds only
python main.py --selftest
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

`tags.py`, no LLM call. The countries in the video are the topic, so the caption
is the hook, the flags that raced, and five hashtags: the winner's country
first, two more of the flags on screen, generic filler after. Which of the
generic ones and in what order is random, so two videos of the same matchup do
not carry identical text.

```bash
python tags.py     # self-check, also what CI runs before rendering
```

[tau]: https://github.com/makiisthenes/TiktokAutoUploader
