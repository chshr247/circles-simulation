"""Flag-circles 9:16 video generator.

Rules: balls spawn with 10-12 lines tethered to the arena wall. They fly and
bounce (off each other and the wall); every bounce speeds them up. A ball
crossing someone else's tether cuts it. Touching the wall grants a new tether.
No tethers left -> the ball dies. Last ball standing wins.
"""
import argparse, colorsys, json, math, os, random, subprocess, sys, time, urllib.parse, urllib.request, wave
import numpy as np
import packs
import tags
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):    # chart titles arrive in alphabets the
    sys.stdout.reconfigure(errors="replace")   # console has no code page for

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1080, 1920, 60   # 60 so the end of the run does not strobe
SUB_MIN, SUB_MAX = 4, 320    # substeps per frame, chosen from the fastest ball
STEP_PX = 6.0                # no ball may move further than this per substep
CX, CY, R = 540, 1080, 500   # arena
BALL_R = 70
LINES_START = 11
MAX_LINES = 24
SLIDE_DEG = 10.0             # anchor drift, deg/s - measured off the reference
ANCHOR_GAP_DEG = 11.5        # spacing anchors keep on the rim; reference sits at 11.7
DECAY_EVERY = 2.0            # sudden death: seconds between forced tether losses
FAN_SPREAD_DEG = 0.0         # arc a ball's tether targets span
DUEL_SPREAD_DEG = 300.0      # ...and once it is down to two, they wrap the rim
SLOWMO_TO = 0.22             # how far time is slowed for the kill
SLOWMO_SECS = 0.5            # seconds of the run, right before the kill
HOLD_S = 1.6                 # freeze on the winner before the file ends
ZOOM_MAX = 1.9               # how far the camera pushes in over the slow finish
ZOOM_TAU = 0.45              # seconds of easing - both for zoom and for the pan
SPAWN_R = 0.85               # spawn ring, as a fraction of the arena radius
CUT_EVERY = 0.0              # 0 = a touched tether is always cut, no recharge
SPEED0, SPEED_GAIN = 400.0, 1.06     # per ricochet of that ball - wall or ball
SPEED_MAX = 6000.0                   # safety net, not a design knob
EARLY_SPEED_MAX = 2000.0             # ...and a real ceiling until it is a duel
BOUNCE_COOLDOWN = 0.05               # min seconds between two banked ricochets
BOUNCE_JITTER_MIN, BOUNCE_JITTER_MAX = 10.0, 25.0   # deflection on every bounce
ESCALATE_EVERY = 3.0                 # seconds between speed step-ups
ESCALATE_PCT = 5.0                   # ...and how much each step adds
DEATH_MOVE = 0.5             # after the kill: seconds still rolling, easing to a stop
CRACK_SECS = 0.45            # crack spreads over DEATH_MOVE + this, ending as it vanishes
REWARD_FULL, REWARD_ZERO = 25.0, 45.0   # wall reward fades out over this window;
                                        # past REWARD_ZERO cuts destroy, not steal
                                        # past REWARD_ZERO cuts destroy, not steal
HARD_STOP = 120.0
MIN_DUR, MAX_DUR = 30.0, 88.0
STATS_TOP = 5                # live leaderboard rows under the arena
STATS_Y, STATS_ROW = 1608, 58
SS = 2                       # supersampling for anti-aliasing
DEFAULT_CC = "random8"


# ---------------------------------------------------------------- geometry
def seg_dist(px, py, ax, ay, bx, by):
    """Distance from point p to segment ab."""
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _clip_end(ax, ay, ox, oy, clip):
    """The tether's open end: the anchor-to-owner segment, stopped `clip`
    short of the owner. None when the whole tether is that short."""
    dx, dy = ox - ax, oy - ay
    L = math.hypot(dx, dy)
    if L <= clip:
        return None, None
    f = (L - clip) / L
    return ax + dx * f, ay + dy * f


def anchor_xy(a):
    """A tether stores only its angle on the rim - anchors drift, so a stored
    point would be stale the moment one moved."""
    return CX + R * math.cos(a), CY + R * math.sin(a)


def _norm(a):
    return a % math.tau


# ---------------------------------------------------------------- simulation
class Ball:
    __slots__ = ("i", "x", "y", "vx", "vy", "r", "lines", "alive", "spd", "ready", "bounces", "next_bounce", "crack")

    def __init__(self, i, x, y, vx, vy, lines):
        self.i, self.x, self.y, self.vx, self.vy = i, x, y, vx, vy
        self.r, self.lines, self.alive = BALL_R, lines, True
        self.spd = SPEED0                  # only ever grows - bounces never slow a ball
        self.ready = 0.0                   # next time this ball may cut again
        self.bounces = 0                   # any ricochet - the only thing that speeds a ball up
        self.next_bounce = 0.0             # see _ricochet
        self.crack = -1.0                  # -1 alive, 0..1 breaking apart


def _jitter(rnd, dx, dy, nx, ny):
    """Scatter a bounce by a few degrees. Perfect reflection makes balls fall
    into repeating orbits; a real one is never that clean. Never rotated back
    into the surface it is leaving, or the ball would burrow into it."""
    a = math.radians(rnd.uniform(BOUNCE_JITTER_MIN, BOUNCE_JITTER_MAX)
                     * rnd.choice((-1.0, 1.0)))
    ca, sa = math.cos(a), math.sin(a)
    rx, ry = dx * ca - dy * sa, dx * sa + dy * ca
    return (rx, ry) if rx * nx + ry * ny > 0 else (dx, dy)


def _aim(b, dx=0.0, dy=0.0):
    """Point the ball along (dx, dy) at its current speed. A bounce changes
    direction and nothing else - swapping momentum would let a head-on hit
    leave a ball nearly stopped, which reads on screen as a speed reset."""
    L = math.hypot(dx, dy)
    if L < 1e-9:
        dx, dy, L = b.vx, b.vy, math.hypot(b.vx, b.vy) or 1.0
    b.vx, b.vy = dx / L * b.spd, dy / L * b.spd


def _ricochet(b, t, cap):
    """A ball's speed comes from its own bounce tally - wall or ball alike.

    The cooldown is not cosmetic. A ball wedged against the wall or between two
    others resolves a bounce on every substep, and at +6% each that is
    1.06**300: one run in twelve pinned the 12000 ceiling while six balls were
    still alive. No real ricochet comes round faster than this.
    """
    if t < b.next_bounce:
        return
    b.next_bounce = t + BOUNCE_COOLDOWN
    b.bounces += 1
    # Speed accumulates instead of being recomputed from the bounce tally. It
    # used to be re-derived every frame and clipped by whatever ceiling was in
    # force, so the moment the duel lifted the ceiling a ball snapped straight
    # to its uncapped value - a measured 2000 to 6000 px/s inside three frames.
    # Raising the ceiling now only allows further growth.
    b.spd = min(cap, b.spd * SPEED_GAIN)


def slide_anchors(balls, dt):
    """Anchors drift towards their owner and shove each other aside so no two
    share a spot. Once per frame: they crawl next to how fast the balls fly,
    and the reference drifts them at only ~10 deg/s anyway."""
    all_ = [[b, k, _norm(a)] for b in balls if b.alive
            for k, a in enumerate(b.lines)]
    if not all_:
        return
    duel = sum(1 for b in balls if b.alive) <= 2
    # Every tether aims at its own slot in an arc around the owner. All of them
    # chasing the single nearest point is what made the duel two thin brooms.
    # In the duel the arc opens right up, and the drift rate goes with it -
    # 300 degrees at 10 deg/s would take half a minute to get there.
    spread = math.radians(DUEL_SPREAD_DEG if duel else FAN_SPREAD_DEG)
    step = math.radians(SLIDE_DEG) * (6.0 if duel else 1.0) * dt
    per_ball = {}
    for it in all_:
        per_ball.setdefault(it[0].i, []).append(it)
    for group in per_ball.values():
        group.sort(key=lambda it: it[2])
        b, m = group[0][0], len(group)
        mid = math.atan2(b.y - CY, b.x - CX)
        for k, it in enumerate(group):
            want = mid + spread * ((k + 0.5) / m - 0.5) if m > 1 else mid
            d = _norm(want - it[2])
            if d > math.pi:
                d -= math.tau                              # shortest way round
            it[2] = _norm(it[2] + max(-step, min(step, d)))
    all_.sort(key=lambda it: it[2])
    n = len(all_)
    sep = min(math.radians(ANCHOR_GAP_DEG), math.tau / (n + 1))
    # Cut the ring at its widest empty stretch and it becomes a plain line to
    # space out. Pushing pairs apart in a loop never converged - every push
    # re-ordered the ring behind it.
    cut, widest = 0, -1.0
    for i in range(n):
        d = (all_[i + 1][2] - all_[i][2]) if i + 1 < n else (all_[0][2] + math.tau - all_[i][2])
        if d > widest:
            widest, cut = d, (i + 1) % n
    base = all_[cut][2]
    pos = [_norm(all_[(cut + k) % n][2] - base) for k in range(n)]
    for k in range(1, n):
        pos[k] = max(pos[k], pos[k - 1] + sep)
    if pos[-1] > math.tau - sep:                           # overflowed the cut
        pos[-1] = math.tau - sep
        for k in range(n - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - sep)
    for k in range(n):
        b, idx, _ = all_[(cut + k) % n]
        b.lines[idx] = _norm(base + pos[k])


def _reward_p(t):
    # ponytail: this linear fade-out is the whole endgame pressure; make it
    # per-ball only if runs stop landing inside MIN_DUR..MAX_DUR.
    if t <= REWARD_FULL:
        return 1.0
    if t >= REWARD_ZERO:
        return 0.0
    return 1.0 - (t - REWARD_FULL) / (REWARD_ZERO - REWARD_FULL)


def _spawn(rnd, n):
    """Evenly spaced on a ring, each with its own wedge of wall - so nobody
    starts sitting inside someone else's fan."""
    balls = []
    off = rnd.uniform(0, math.tau)
    for i in range(n):
        a = off + i * math.tau / n + rnd.uniform(-0.15, 0.15)
        rr = R * SPAWN_R
        x, y = CX + rr * math.cos(a), CY + rr * math.sin(a)
        d = rnd.uniform(0, math.tau)
        share = math.tau / n
        lines = [_norm(a + (k - (LINES_START - 1) / 2) * share / (LINES_START + 1))
                 for k in range(LINES_START)]
        balls.append(Ball(i, x, y, SPEED0 * math.cos(d), SPEED0 * math.sin(d), lines))
    return balls


def simulate(seed, n_balls, record=True):
    rnd = random.Random(seed)
    balls = _spawn(rnd, n_balls)
    frames, events = [], []
    t, next_decay = 0.0, 0.0
    ending, doomed, cracked = None, None, False
    esc_step = 0
    duel_at = None
    while t < HARD_STOP:
        # the faster the field, the finer the step - otherwise a ball at
        # several thousand px/s steps clean over tethers and other balls
        vmax = max((b.spd for b in balls if b.alive), default=SPEED0)
        sub = max(SUB_MIN, min(SUB_MAX, int(vmax / FPS / STEP_PX) + 1))
        h = 1.0 / FPS / sub
        for _ in range(sub):
            alive = [b for b in balls if b.alive]
            cap = EARLY_SPEED_MAX if len(alive) > 2 else SPEED_MAX
            # Once the last tether is cut the pair keeps rolling for a moment
            # and coasts to a stop, so the kill can be watched instead of just
            # being over.
            roll = 1.0 if ending is None else max(0.0, 1.0 - (t - ending) / DEATH_MOVE)
            for b in alive:                                    # move + wall
                b.x += b.vx * h * roll
                b.y += b.vy * h * roll
                d = math.hypot(b.x - CX, b.y - CY) or 1e-9
                if d + b.r > R:
                    nx, ny = (b.x - CX) / d, (b.y - CY) / d
                    b.x, b.y = CX + nx * (R - b.r), CY + ny * (R - b.r)
                    vn = b.vx * nx + b.vy * ny
                    _ricochet(b, t, cap)
                    _aim(b, *_jitter(rnd, b.vx - 2 * vn * nx, b.vy - 2 * vn * ny,
                                     -nx, -ny))
                    events.append(("wall", t))
                    if ending is None and len(b.lines) < MAX_LINES and rnd.random() < _reward_p(t):
                        b.lines.append(_norm(math.atan2(ny, nx)))
            for i in range(len(alive)):                        # ball vs ball
                for j in range(i + 1, len(alive)):
                    a, c = alive[i], alive[j]
                    dx, dy = c.x - a.x, c.y - a.y
                    d = math.hypot(dx, dy) or 1e-9
                    if d < a.r + c.r:
                        nx, ny = dx / d, dy / d
                        ov = (a.r + c.r - d) / 2
                        a.x -= nx * ov
                        a.y -= ny * ov
                        c.x += nx * ov
                        c.y += ny * ov
                        van = a.vx * nx + a.vy * ny
                        vcn = c.vx * nx + c.vy * ny
                        if van - vcn > 0:                      # only if approaching
                            adx, ady = a.vx + (vcn - van) * nx, a.vy + (vcn - van) * ny
                            cdx, cdy = c.vx + (van - vcn) * nx, c.vy + (van - vcn) * ny
                            _ricochet(a, t, cap)
                            _ricochet(c, t, cap)
                            _aim(a, *_jitter(rnd, adx or -nx, ady or -ny, -nx, -ny))
                            _aim(c, *_jitter(rnd, cdx or nx, cdy or ny, nx, ny))
                            events.append(("hit", t))
            for b in alive:                                    # cutting tethers
                if ending is not None:
                    break                                      # the match is decided
                if b.ready > t:
                    continue
                for o in alive:
                    if o is b or not o.lines:
                        continue
                    # Every tether the ball touches is cut - but only on its
                    # open span. Right at the owner all its tethers converge
                    # tighter than a ball is wide, and that stretch is hidden
                    # behind the owner anyway, so one bump would otherwise
                    # shear a whole fan at once and kill on first contact.
                    clip = o.r + b.r
                    keep, taken = [], []
                    for ang in o.lines:
                        ax, ay = anchor_xy(ang)
                        ex, ey = _clip_end(ax, ay, o.x, o.y, clip)
                        if ex is None or seg_dist(b.x, b.y, ax, ay, ex, ey) >= b.r:
                            keep.append(ang)
                        else:
                            taken.append(ang)
                    if taken:
                        events.append(("cut", t))
                        o.lines = keep
                        # The cut tether changes hands where it stands, then
                        # drifts over to its new owner. Past the reward window
                        # it just snaps: tethers that only ever change owner are
                        # conserved, and two even balls would trade forever.
                        if t < REWARD_ZERO:
                            for ang in taken:
                                if len(b.lines) < MAX_LINES:
                                    b.lines.append(ang)
                        b.ready = t + CUT_EVERY
                        if not o.lines:
                            events.append(("die", t))
                            if sum(1 for x in balls if x.alive and x is not o) == 1:
                                ending, doomed = t, o          # hold it for the crack
                                o.crack = 0.0
                            else:
                                o.alive = False
                        break
            t += h
        live = [b for b in balls if b.alive]
        cap = EARLY_SPEED_MAX if len(live) > 2 else SPEED_MAX
        step_now = int(t / ESCALATE_EVERY)
        if step_now > esc_step:                       # the standing bonus, on a timer
            esc_step = step_now
            for b in live:
                b.spd = min(cap, b.spd * (1.0 + ESCALATE_PCT / 100.0))
        slide_anchors(balls, 1.0 / FPS)
        if t > REWARD_ZERO and t > next_decay:
            # Fans hug their owner, so two balls can genuinely never meet and
            # the run would sit there forever. Past the window everyone starts
            # shedding tethers whether or not anyone cuts them.
            next_decay = t + DECAY_EVERY
            live = [b for b in balls if b.alive]
            lead = max(live, key=lambda b: (len(b.lines), -b.i), default=None)
            for b in live:                       # the leader is spared, so the
                if b is lead or not b.lines:     # last tether can never go from
                    continue                     # everyone at once - no draws
                b.lines.pop()
                if not b.lines:
                    events.append(("die", t))
                    if sum(1 for x in balls if x.alive and x is not b) == 1:
                        ending, doomed = t, b
                        b.crack = 0.0
                    else:
                        b.alive = False
        if ending is not None:
            doomed.crack = min(1.0, (t - ending) / (DEATH_MOVE + CRACK_SECS))
            if not cracked:
                cracked = True
                events.append(("crack", ending))
        if record:
            frames.append([(b.x, b.y, b.i, tuple(b.lines), b.crack)
                           for b in balls if b.alive])
        if ending is not None and t - ending >= DEATH_MOVE + CRACK_SECS:
            doomed.alive = False
            events.append(("gone", t))
            if record:                       # one clean frame without it, or the
                frames.append([(b.x, b.y, b.i, tuple(b.lines), b.crack)
                               for b in balls if b.alive])   # hold freezes the corpse
        alive = [b for b in balls if b.alive]
        if duel_at is None and len(alive) <= 2:
            duel_at = t
            events.append(("duel", t))
        if len(alive) <= 1:
            return dict(frames=frames, events=events, duration=t,
                        winner=alive[0].i if alive else None,
                        runner=doomed.i if doomed else None,
                        duel=t - (duel_at if duel_at is not None else t))
    return None


def slow_finish(run):
    """Stretch the last stretch of the run into slow motion.

    Done to the finished timeline rather than inside the physics on purpose:
    slowing time while the sim runs changes the integration step, so the run
    would come out different - and, worse, live code cannot know when the kill
    lands, which had slow motion firing seven seconds early and expiring
    before the last tether went. Here the ending is already known.
    """
    frames, events = run["frames"], run["events"]
    n = int((SLOWMO_SECS + DEATH_MOVE + CRACK_SECS) * FPS)
    if len(frames) < n + 2:
        return run
    steps = max(1, int(round(1 / SLOWMO_TO)))
    t0 = (len(frames) - n) / FPS
    head, tail = frames[:-n], frames[-n:]

    out = list(head)
    for i in range(len(tail) - 1):
        a = {row[2]: row for row in tail[i]}
        b = {row[2]: row for row in tail[i + 1]}
        for st in range(steps):
            f = st / steps
            out.append([(a[k][0] + (b[k][0] - a[k][0]) * f,
                         a[k][1] + (b[k][1] - a[k][1]) * f,
                         k, b[k][3],
                         a[k][4] + (b[k][4] - a[k][4]) * f) for k in a if k in b])
    out.append(tail[-1])
    for _ in range(int(HOLD_S * FPS)):            # hold on the winner
        out.append(out[-1])

    # Stretched by whole frames, so the soundtrack has to use the same integer
    # factor - dividing by SLOWMO_TO instead drifted the glass half a second
    # ahead of the picture by the end.
    ev = [(k, tt if tt <= t0 else t0 + (tt - t0) * steps) for k, tt in events]
    ev.append(("slow", t0))
    run["frames"], run["events"] = out, ev
    run["camera"] = _camera(out, len(head), run["winner"])
    run["duration"] = len(out) / FPS
    return run


def _camera(frames, from_frame, winner=None):
    """One (x, y, zoom) per frame. The push-in starts with the slow motion and
    stays on the winner, eased so it drifts rather than snaps."""
    a = 1.0 - math.exp(-1.0 / (ZOOM_TAU * FPS))
    cx, cy, z = float(CX), float(CY), 1.0
    track = []
    for k, snap in enumerate(frames):
        if k >= from_frame and snap:
            row = next((r for r in snap if r[2] == winner), None)
            if row is None:                            # before the kill is settled
                row = (sum(r[0] for r in snap) / len(snap),
                       sum(r[1] for r in snap) / len(snap))
            tx, ty, tz = row[0], row[1], ZOOM_MAX
            # Stay on the winner, but back off just enough to keep the loser in
            # shot while it is still cracking - at full zoom the view is 568 px
            # across against a 1000 px arena, and a kill on the far side put it
            # off-screen in one run out of five.
            others = [r for r in snap if r[2] != winner]
            if others:
                dx = max(abs(r[0] - tx) for r in others) + BALL_R + 30
                dy = max(abs(r[1] - ty) for r in others) + BALL_R + 30
                tz = min(tz, max(1.0, min(W / (2 * dx), H / (2 * dy))))
        else:
            tx, ty, tz = float(CX), float(CY), 1.0
        cx += (tx - cx) * a
        cy += (ty - cy) * a
        z += (tz - z) * a
        half_w, half_h = W / (2 * z), H / (2 * z)      # keep the crop on-screen
        track.append((min(max(cx, half_w), W - half_w),
                      min(max(cy, half_h), H - half_h), z))
    return track


def find_seed(seed_arg, n_balls, want=None, pair=None, limit=400, shortlist=8):
    """Search seeds for a run that lands in the duration window.

    With `want` set, only runs that ball wins are considered, and of those the
    one with the longest final duel is taken - that stretch is the whole payoff,
    so a seed that gets there in two seconds makes a dull video.

    `pair` additionally pins who the last two standing are. Out of eight balls
    that is one run in 28 before the winner is even asked for, so the search
    gets a far bigger budget - it is still seconds, the shortlisting sim does
    not record frames.
    """
    if seed_arg != "auto":
        return int(seed_arg)
    if pair:
        limit, shortlist = limit * 10, 3
    start = random.randrange(1 << 30)
    best, tried = [], 0
    for k in range(limit):
        s = start + k
        tried = k + 1
        r = simulate(s, n_balls, record=False)
        if not r or not MIN_DUR <= r["duration"] <= MAX_DUR:
            continue
        if want is not None and r["winner"] != want:
            continue
        if pair and {r["winner"], r["runner"]} != pair:
            continue
        best.append((r["duel"], s, r["duration"], r["winner"]))
        if want is None or len(best) >= shortlist:
            break
    if not best:
        sys.exit("no seed matched - loosen the knobs or pick another winner")
    duel, s, dur, win = max(best)
    print("seed %d: %.1fs, winner #%d, final duel %.1fs (%d tried, %d matched)"
          % (s, dur, win, duel, tried, len(best)))
    return s


# ---------------------------------------------------------------- pictures
def item_image(pack, item):
    return Image.open(packs.image_path(pack, item)).convert("RGB")


def dominant(img):
    """Most common strongly-saturated colour; falls back to the most common one."""
    a = np.asarray(img.resize((60, 40))).reshape(-1, 3).astype(np.int16)
    sat = (a.max(1) - a.min(1)) > 60
    pick = a[sat] if sat.sum() > 40 else a
    q = (pick // 32).astype(np.int32)
    key = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    best = np.bincount(key).argmax()
    c = pick[key == best].mean(0)
    return _neon(c)


def _neon(c):
    """Push the colour to full brightness - flag colours are too muddy to
    carry the tether fans against black."""
    c = np.asarray(c, float)
    c = c * (255.0 / max(c.max(), 1.0))
    c = c + (255.0 - c) * 0.18            # lift so dark blues/greens still read
    return tuple(int(min(255, round(v))) for v in c)


def disc(img, px):
    """Picture cropped into a circle of diameter px, with an alpha mask.

    Centred sideways always, but only centred vertically on a wide picture. A
    flag is wider than tall and a portrait is not, and the middle square of a
    portrait is a chest: faces sit near the top, so a tall picture is cropped
    from near the top or the pack of rappers is a pack of shoulders.
    """
    w, h = img.size
    s = min(w, h)
    top = (h - s) // 2 if w >= h else int((h - s) * 0.12)
    sq = img.crop(((w - s) // 2, top, (w + s) // 2, top + s))
    sq = sq.resize((px, px), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (px * 4, px * 4), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, px * 4 - 1, px * 4 - 1), fill=255)
    sq.putalpha(mask.resize((px, px), Image.LANCZOS))
    return sq


# ---------------------------------------------------------------- audio
# ponytail: fixed tempo. Only the ORDER of the track's notes is used, never its
# timing, so there is no tempo to inherit - this is the grid the run gets snapped
# to. Tune by ear per batch; estimate it off the source track if that ever stops
# being enough.
BPM = 126.0
BEAT = 60.0 / BPM
GRID = BEAT / 4.0            # 1/16 - tight enough that the run still drives it
DUEL_RAMP = 2.5              # seconds the backing takes to come up under a duel

MELODY_SR = 22050
MELODY_N, MELODY_H = 2048, 512


AUDIO_EXT = (".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac")


CHART_API = "https://api.deezer.com/chart/0/tracks?limit=%d"
SEARCH_API = "https://api.deezer.com/search?q=%s&order=RANKING&limit=%d"
UA = "Mozilla/5.0"


def fetch_trending(dest, n=5, query=""):
    """Top tracks off Deezer's chart, straight into the melody folder.

    No key and no account, and the 30-second preview it serves is plenty: only
    the ORDER of the notes is ever taken and none of the audio reaches the
    video. Tracks already in the folder are left alone, so the rotation grows
    rather than churns, and a dead network is a warning, not a lost render.
    """
    url = SEARCH_API % (urllib.parse.quote(query), n) if query else CHART_API % n
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        data = json.load(urllib.request.urlopen(req, timeout=20)).get("data", [])
    except Exception as e:
        print("  trending: %s - оставляю папку как есть" % e)
        return []
    got = []
    for t in data:
        prev = t.get("preview")
        if not prev:
            continue
        name = "%s - %s" % (t["artist"]["name"], t["title"])
        name = "".join(c for c in name if c.isalnum() or c in " -_()").strip()
        path = os.path.join(dest, name[:70] + ".mp3")
        if not os.path.exists(path):
            try:
                r = urllib.request.urlopen(
                    urllib.request.Request(prev, headers={"User-Agent": UA}), timeout=30)
                with open(path, "wb") as f:
                    f.write(r.read())
            except Exception as e:
                print("  trending: %s - %s" % (name, e))
                continue
            print("  + %s (%.1f MB)" % (os.path.basename(path),
                                        os.path.getsize(path) / 1e6))
        got.append(path)
    return got


def melody_pool(path):
    """Every track in a folder, or one named file. The folder is re-read each
    run, so dropping a new track in is all it takes to add it to the rotation."""
    if os.path.isdir(path):
        return sorted(os.path.join(path, f) for f in os.listdir(path)
                      if f.lower().endswith(AUDIO_EXT))
    return [path] if os.path.exists(path) else []


def melody_notes(path):
    """The ordered pitch sequence of a track, as MIDI numbers.

    Onset detection by spectral flux, then a harmonic product spectrum on each
    onset - it reports the fundamental rather than the loudest partial, which
    matters when an 808 is sitting on top of everything. Only the order of the
    notes is taken; none of the audio is used. Cached beside the file, the
    analysis takes about a second for a three-minute track.
    """
    cdir = os.path.join(HERE, "sounds", ".notes")
    os.makedirs(cdir, exist_ok=True)
    cache = os.path.join(cdir, os.path.basename(path) + ".json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)

    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
                          "-ac", "1", "-ar", str(MELODY_SR), "-"],
                         stdout=subprocess.PIPE).stdout
    x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if len(x) < MELODY_N * 4:
        return []

    win = np.hanning(MELODY_N).astype(np.float32)
    steps = 1 + (len(x) - MELODY_N) // MELODY_H
    mag = np.empty((steps, MELODY_N // 2 + 1), np.float32)
    for i in range(steps):
        mag[i] = np.abs(np.fft.rfft(x[i * MELODY_H:i * MELODY_H + MELODY_N] * win))

    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(1)
    flux /= (flux.max() or 1.0)
    k = 16
    pad = np.pad(flux, (k, k), mode="edge")
    local = np.array([pad[i:i + 2 * k + 1].mean() for i in range(len(flux))])
    thr = local * 1.6 + 0.02

    notes, last = [], -99
    freqs = np.fft.rfftfreq(8192, 1.0 / MELODY_SR)
    for i in range(1, len(flux) - 1):
        if not (flux[i] > thr[i] and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]):
            continue
        if i - last < 4:
            continue
        last = i
        seg = x[i * MELODY_H:i * MELODY_H + 4096]
        if len(seg) < 2048:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8192))
        hps = spec[:len(spec) // 4].copy()
        for h in (2, 3, 4):
            hps *= spec[::h][:len(hps)]
        band = freqs[:len(hps)]
        keep = (band >= 70.0) & (band <= 1400.0)
        if not keep.any():
            continue
        f = float(band[np.argmax(np.where(keep, hps, 0.0))])
        if not 70.0 <= f <= 1400.0:
            continue
        midi = int(round(69 + 12 * math.log2(f / 440.0)))
        if 36 <= midi <= 96:
            notes.append(midi)

    if notes:
        # Transposed by whole octaves so a bare tone can be heard over the rest.
        # Nothing else is touched: folding stray octaves back looked tidy on
        # paper but inverted the melody's own leaps, which is the one thing
        # worth keeping.
        mid = sorted(notes)[len(notes) // 2]
        shift = 12 * int(round((62 - mid) / 12.0))
        notes = [n + shift for n in notes]
    with open(cache, "w") as f:
        json.dump(notes, f)
    return notes


def load_sound(name, secs=None, peak=0.9, sr=44100):
    """Decode one of the samples in sounds/ through ffmpeg - it is already a
    dependency, and this saves pulling in an audio library for two files."""
    cmd = ["ffmpeg", "-v", "error", "-i", os.path.join(HERE, "sounds", name)]
    if secs:
        cmd += ["-t", str(secs)]
    cmd += ["-f", "s16le", "-ac", "1", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    if len(a) == 0:
        return np.zeros(1, np.float32)
    tail = min(len(a), int(sr * 0.05))          # a trim without a fade clicks
    a = a.copy()
    a[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)
    return a * (peak / max(float(np.abs(a).max()), 1e-6))


def synth(events, dur, notes=None, sr=44100):
    buf = np.zeros(int((dur + 3.5) * sr), np.float32)
    # Straight from the source track when one is given, so the cuts play its
    # melody in its own order; a bare pentatonic otherwise.
    scale = ([440.0 * 2 ** ((n - 69) / 12.0) for n in notes] if notes else
             [261.63, 293.66, 329.63, 392.00, 440.00, 523.25, 587.33, 659.25])
    duel_t = next((t for k, t in events if k == "duel"), None)

    def heat(t):
        """0 while the field is crowded, 1 once the duel has settled in."""
        if duel_t is None:
            return 0.0
        return min(1.0, max(0.0, (t - duel_t) / DUEL_RAMP))

    def env(n, k):
        e = np.exp(-np.arange(n) / sr * k).astype(np.float32)
        a = min(n, int(sr * 0.004))     # without the ramp every note starts on a click
        e[:a] *= np.linspace(0.0, 1.0, a, dtype=np.float32)
        return e

    def tone(f, ms, k, amp, harm=0.35, detune=0.0):
        n = int(sr * ms / 1000)
        t = np.arange(n) / sr
        w = np.sin(2 * math.pi * f * t) + harm * np.sin(4 * math.pi * f * t)
        if detune:                      # a second, barely off oscillator - a lone
            w = w + detune * np.sin(2 * math.pi * f * 1.004 * t)   # sine reads as a test tone
        return (w * env(n, k) * amp).astype(np.float32)

    def mix(s, t0):
        i = int(t0 * sr)
        m = min(len(s), len(buf) - i)
        if m > 0:
            buf[i:i + m] += s[:m]

    wall = tone(1500, 55, 60, 0.30, 0.25)

    def slide_tone(f0, f1, ms, k, amp, harm=0.0):
        n = int(sr * ms / 1000)
        t = np.arange(n) / sr
        T = ms / 1000.0
        ph = 2 * math.pi * f0 * T / math.log(f1 / f0) * ((f1 / f0) ** (t / T) - 1)
        w = np.sin(ph) + harm * np.sin(2 * ph)
        return (w * env(n, k) * amp).astype(np.float32)

    die = slide_tone(220, 45, 600, 6, 0.85, 0.5)       # a ball loses its last tether

    crack = load_sound("impact.mp3", peak=0.95)        # the moment it splits
    gone = load_sound("web.mp3", secs=1.0, peak=0.95)  # ...and when it goes
    slow = slide_tone(700, 90, 1100, 2.2, 0.36, 0.3)   # time easing down

    thud = tone(90, 260, 9, 0.85, 0.6)                 # ball on ball
    nn = int(sr * 0.07)                                # a tether parting
    ar = np.arange(nn)
    snip = (np.sin(2 * math.pi * (2200 - 14000 * ar / sr) * ar / sr)
            * env(nn, 60) * 0.35).astype(np.float32)

    # The melody rides whichever event this ruleset actually produces most.
    # Hanging it on a rare one leaves the track silent - that already happened
    # once, when it sat on ball-on-ball hits at ten a run.
    seen = {}
    for kind, _t in events:
        if kind in ("cut", "wall", "hit"):
            seen[kind] = seen.get(kind, 0) + 1
    lead = max(seen, key=seen.get) if seen else "cut"
    voice = {"wall": wall, "hit": thud, "cut": snip}
    # These constants fire ~90 cuts a second. One note each is not a melody, it
    # is a buzzsaw, so the tune only takes the first cut in each 1/16 slot. The
    # grid is the whole trick: the same chaos, landed on a beat, reads as played
    # rather than spilled. Shifts a note by at most 30ms, which nobody sees.
    # Deaths and the crack are never snapped and never dropped.
    slot = -1.0
    k = 0
    for kind, t0 in events:
        if kind == lead:
            # Cuts saturate the 1/16 grid end to end, so in the duel the grid
            # halves: with the notes twice as long, keeping every slot stacks
            # them into mud instead of a line.
            g = GRID * 2 if heat(t0) > 0.5 else GRID
            q = round(t0 / g) * g
            if q <= slot:
                continue
            slot, t0 = q, q
        if kind == "die":
            s = die
        elif kind == "crack":
            s = crack
        elif kind == "gone":
            s = gone
        elif kind == "slow":
            s = slow
        elif kind == "duel":
            continue                    # the backing below is its voice
        elif kind == lead:
            h = heat(t0)
            # In the duel the note is let off the leash: longer, warmer, wider.
            # Short blips read as chatter, notes that ring read as a line.
            s = tone(scale[k % len(scale)], 240 + 280 * h, 12 - 8 * h,
                     0.62, 0.35 + 0.25 * h, 0.3 * h)
            k += 1
        else:
            s = voice.get(kind)
            if s is None:
                continue
        mix(s, t0)

    # Once it is down to two, events alone stop carrying a tune - so a backing
    # comes up under them: root on the beat, a chord on the bar. Ramped in over
    # DUEL_RAMP, so it grows in as the field thins instead of switching on.
    if duel_t is not None:
        f0 = sorted(scale)[len(scale) // 2]        # the track's central pitch
        while f0 > 110.0:                          # ...dropped to where a bass lives
            f0 /= 2.0
        t = math.ceil(duel_t / BEAT) * BEAT        # on the same grid as the notes
        beat = 0
        while t < dur:
            g = heat(t)
            f = f0 if (beat // 8) % 2 == 0 else f0 * 1.5      # tonic, then the fifth
            mix(tone(f, 420, 5.0, 0.50 * g, 0.5), t)
            if beat % 4 == 0:                                 # chord on the bar
                for m in (2.0, 3.0, 4.0):
                    mix(tone(f * m, 950, 1.2, 0.11 * g, 0.15, 0.3), t)
            beat += 1
            t += BEAT

    # A room, opening up as the duel takes over. Feed-forward taps only: dry
    # blips sound like a signal generator, the tail is what makes them sound
    # played, and five slices do it without a filter library.
    wet = np.zeros_like(buf)
    for d, g in ((0.037, 0.50), (0.061, 0.38), (0.089, 0.30),
                 (0.131, 0.22), (0.187, 0.15)):
        n = int(d * sr)
        wet[n:] += buf[:-n] * g
    wet = np.convolve(wet, np.ones(24, np.float32) / 24, "same")   # dulls the tail
    if duel_t is None:
        buf += wet * 0.15
    else:
        tt = np.arange(len(buf), dtype=np.float32) / sr
        buf += wet * (0.15 + 0.35 * np.clip((tt - duel_t) / DUEL_RAMP, 0.0, 1.0))

    peak = float(np.abs(buf).max()) or 1.0
    return np.clip(buf / max(1.0, peak / 0.9), -1, 1)


def write_wav(path, buf, sr=44100):
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((buf * 32767).astype("<i2").tobytes())


# ---------------------------------------------------------------- rendering
def font(px):
    # The last one is what a Linux runner has; without it PIL falls through to
    # a bitmap default and the hook renders at about six pixels.
    for name in ("ariblk.ttf", "arialbd.ttf", "impact.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            pass
    return ImageFont.load_default()


def spread_hues(cols, min_gap=34.0, budget=60.0):
    """Half the world's flags are red-dominant, so several fans come out the
    same colour. Fan out each crowded cluster around its own centre - crimson
    vs orange-red still reads as the red country's line, but the two are now
    tellable apart."""
    hsv = [list(colorsys.rgb_to_hsv(*(v / 255.0 for v in c))) for c in cols]
    order = sorted(range(len(hsv)), key=lambda k: hsv[k][0])
    hues = [hsv[k][0] * 360.0 for k in order]

    clusters, cur = [], [0]                       # consecutive hues that crowd
    for i in range(1, len(hues)):
        if hues[i] - hues[i - 1] < min_gap:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    if len(clusters) > 1 and (hues[0] + 360 - hues[-1]) < min_gap:
        clusters[0] = clusters.pop() + clusters[0]   # the set wraps past 360

    out = [None] * len(cols)
    for cl in clusters:
        m = len(cl)
        centre = hues[cl[0]] + ((hues[cl[-1]] - hues[cl[0]]) % 360) / 2
        step = min(min_gap, 2 * budget / (m - 1)) if m > 1 else 0.0
        for k, idx in enumerate(cl):
            h = (centre + (k - (m - 1) / 2) * step) % 360
            _, s_, _v = hsv[order[idx]]
            r, g, b = colorsys.hsv_to_rgb(h / 360.0, max(s_, 0.55), 1.0)
            out[order[idx]] = tuple(int(round(x * 255)) for x in (r, g, b))
    return out


def _crack_lines(cx, cy, r, seed_i, grow):
    """Splinters spreading out of the ball's centre as it breaks up. Angles are
    derived from the ball index so the same ball always cracks the same way."""
    out = []
    for k in range(5):
        a0 = seed_i * 2.399 + k * 1.2566
        pts, ang = [(cx, cy)], a0
        for step, wig in ((0.35, 0.0), (0.72, 0.22), (1.05, -0.26)):
            ang = a0 + wig
            rr = r * grow * step
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        for j in range(len(pts) - 1):
            out.append([pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1]])
    return out


def draw_stats(img, snap, names, cols, f):
    """Live top-N by tether count, under the arena. Drawn on the cropped frame
    rather than the arena layer, same as the hook: the finish zooms in, and a
    scoreboard that zooms with it walks straight off the bottom of the video."""
    d = ImageDraw.Draw(img)
    rows = sorted(snap, key=lambda r: (-len(r[3]), r[2]))[:STATS_TOP]
    for k, row in enumerate(rows):
        i, n = row[2], len(row[3])
        y = STATS_Y + k * STATS_ROW + STATS_ROW // 2
        d.rectangle([60, y - 13, 78, y + 13], fill=cols[i])
        name = names[i]
        while name and d.textlength(name, font=f) > 830:      # long ones exist
            name = name[:-1]
        d.text((98, y), name, font=f, fill=(255, 255, 255), anchor="lm",
               stroke_width=3, stroke_fill=(0, 0, 0))
        d.text((1020, y), str(n), font=f, fill=cols[i], anchor="rm",
               stroke_width=3, stroke_fill=(0, 0, 0))


def render(run, pack, ids, hook, out, preview=0, notes=None):
    flags = [item_image(pack, i) for i in ids]
    roster = packs.roster(pack)
    names = [packs.display(roster.get(i, i)).upper() for i in ids]
    stats_font = font(38)
    cols = spread_hues([dominant(f) for f in flags])
    discs = [disc(f, BALL_R * 2 * SS) for f in flags]

    base = Image.new("RGB", (W * SS, H * SS), (5, 5, 8))
    d = ImageDraw.Draw(base)
    d.ellipse([(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS],
              outline=(255, 255, 255), width=3 * SS)

    # The hook sits on its own layer, pasted after the camera crop - otherwise
    # the push-in would blow the text up and shove it off the frame.
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if hook:
        od = ImageDraw.Draw(overlay)
        px = 78
        while px > 30:                                # shrink to fit the frame
            f = font(px)
            if od.textlength(hook.upper(), font=f) <= W - 120:
                break
            px -= 3
        od.text((W // 2, 300), hook.upper(), font=f, fill=(255, 255, 255, 255),
                anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0, 255))

    frames = run["frames"][:preview * FPS] if preview else run["frames"]
    cam = run.get("camera") or [(W / 2.0, H / 2.0, 1.0)] * len(frames)
    dur = len(frames) / FPS
    os.makedirs(os.path.dirname(out), exist_ok=True)
    wav = os.path.splitext(out)[0] + ".wav"
    heard = [e for e in run["events"] if e[1] < dur]
    tally = {}
    for kind, _t in heard:
        if kind in ("cut", "wall", "hit"):
            tally[kind] = tally.get(kind, 0) + 1
    if tally:
        ru = {"cut": "срез", "wall": "стенка", "hit": "столкновение"}
        order = sorted(tally.items(), key=lambda kv: -kv[1])
        print("  мелодия на: %s   (%s)" % (ru[order[0][0]],
              ", ".join("%s %d" % (ru[k], v) for k, v in order)))
    write_wav(wav, synth(heard, dur, notes))

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k, snap in enumerate(frames):
        img = base.copy()
        dr = ImageDraw.Draw(img)
        for x, y, i, lines, crack in snap:
            for ang in lines:
                ax, ay = anchor_xy(ang)
                dr.line([ax * SS, ay * SS, x * SS, y * SS], fill=cols[i], width=3 * SS)
        for x, y, i, lines, crack in snap:
            px, py = int(x * SS) - BALL_R * SS, int(y * SS) - BALL_R * SS
            img.paste(discs[i], (px, py), discs[i])
            dr.ellipse([px, py, px + BALL_R * 2 * SS, py + BALL_R * 2 * SS],
                       outline=cols[i], width=3 * SS)
            if crack > 0:
                for seg in _crack_lines(x * SS, y * SS, BALL_R * SS, i, crack):
                    dr.line(seg, fill=(0, 0, 0), width=3 * SS)
        cx, cy, z = cam[k] if k < len(cam) else (W / 2.0, H / 2.0, 1.0)
        if z > 1.001:
            box = ((cx - W / (2 * z)) * SS, (cy - H / (2 * z)) * SS,
                   (cx + W / (2 * z)) * SS, (cy + H / (2 * z)) * SS)
            shot = img.resize((W, H), Image.LANCZOS, box=box)
        else:
            shot = img.resize((W, H), Image.LANCZOS)
        shot.paste(overlay, (0, 0), overlay)
        draw_stats(shot, snap, names, cols, stats_font)
        p.stdin.write(shot.tobytes())
        if k % 150 == 0:
            print("  frame %d/%d" % (k, len(frames)), flush=True)
    p.stdin.close()
    p.wait()
    os.remove(wav)
    return out


# ---------------------------------------------------------------- self-check
def selftest():
    assert abs(seg_dist(0, 5, -10, 0, 10, 0) - 5) < 1e-9
    assert abs(seg_dist(20, 0, -10, 0, 10, 0) - 10) < 1e-9      # past the end cap
    b = Ball(0, 0, 0, 30.0, 40.0, [(1, 1)])         # a bounce never slows a ball
    for direction in ((-1, 0), (0, 1), (0.3, -0.4)):
        _aim(b, *direction)
        assert abs(math.hypot(b.vx, b.vy) - b.spd) < 1e-9, "a bounce changed speed"
    was = b.spd
    _ricochet(b, 0.0, SPEED_MAX)
    assert b.spd > was and b.bounces == 1, "a ricochet must speed the ball up"
    _ricochet(b, BOUNCE_COOLDOWN / 2, SPEED_MAX)          # wedged, not ricocheting
    assert b.bounces == 1, "the bounce cooldown let a stuck ball bank speed"
    _ricochet(b, BOUNCE_COOLDOWN * 2, 1.0)
    assert b.spd <= 1.0, "the speed ceiling was ignored"
    ring = [Ball(k, 0, 0, 1.0, 0.0, [0.0, 0.01, 0.02]) for k in range(3)]
    slide_anchors(ring, 1.0)                        # anchors must never overlap
    angs = sorted(a for x in ring for a in x.lines)
    gaps = [angs[k + 1] - angs[k] for k in range(len(angs) - 1)]
    gaps.append(angs[0] + math.tau - angs[-1])
    want = min(math.radians(ANCHOR_GAP_DEG), math.tau / (len(angs) + 1))
    assert min(gaps) > want - 1e-9, "anchors ended up on top of each other"
    ok, durs = 0, []
    for s in range(12345, 12365):
        r = simulate(s, 8, record=False)
        if r:
            ok += 1
            durs.append(round(r["duration"], 1))
            assert r["winner"] is not None and 0 < r["duration"] <= HARD_STOP
            assert r["runner"] is not None and r["runner"] != r["winner"]
    print("selftest ok: %d/20 runs finished, durations %s" % (ok, sorted(durs)))
    # the leaderboard is sorted by tether count, most first, ties by ball index
    snap = [(0, 0, 3, (1,) * 4, -1.0), (0, 0, 1, (1,) * 9, -1.0), (0, 0, 0, (1,) * 9, -1.0)]
    assert [r[2] for r in sorted(snap, key=lambda r: (-len(r[3]), r[2]))] == [0, 1, 3]


def one(pack, ids, hook, seed_arg, out, preview, notes=None, want=None, pair=None):
    seed = find_seed(seed_arg, len(ids), want, pair)
    run = slow_finish(simulate(seed, len(ids)))
    winner = ids[run["winner"]]
    print("winner: %s  duration: %.1fs  events: %d"
          % (winner.upper(), run["duration"], len(run["events"])))
    render(run, pack, ids, hook, out, preview, notes)
    # Written beside the mp4 for whoever posts it. Nothing here can be worked
    # out from the file later: the countries are drawn at random and the winner
    # is only known once the run has been simulated.
    with open(os.path.splitext(out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"hook": hook, "pack": pack, "items": ids, "winner": winner,
                   "seed": seed, "caption": tags.caption(hook, pack, ids, winner)}, f,
                  ensure_ascii=False, indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default="WHO WILL WIN",
                    help="text on the frame, or \"random\" for one per video")
    ap.add_argument("--pack", default=packs.DEFAULT,
                    help="what races: %s, or \"random\" for one per video"
                         % ", ".join(packs.all_packs()))
    ap.add_argument("--countries", default=DEFAULT_CC,
                    help="ids from the pack, or random8 / random10 to draw from it")
    ap.add_argument("--seed", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--count", type=int, default=1, help="how many videos to render")
    ap.add_argument("--winner", default="", help="the id that must win")
    ap.add_argument("--finalists", default="",
                    help="the two ids that must be the last standing, e.g. ua,us")
    ap.add_argument("--preview", type=int, default=0, help="render only first N seconds")
    ap.add_argument("--melody", default=os.path.join(HERE, "sounds", "music"),
                    help="track, or a folder of them to draw from; \"\" for a plain scale")
    ap.add_argument("--trending", type=int, default=0,
                    help="pull N chart previews into the melody folder first")
    ap.add_argument("--trending-q", default="",
                    help="which chart, e.g. \"phonk\"; blank = global top")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.trending and os.path.isdir(a.melody):
        print("trending: %d из %s" % (a.trending, a.trending_q or "мирового чарта"))
        fetch_trending(a.melody, a.trending, a.trending_q)
    pool = melody_pool(a.melody) if a.melody else []
    for k in range(a.count):
        notes = None
        if pool:                                   # a different track each video
            track = random.choice(pool)
            notes = melody_notes(track)
            print("melody: %s (%d notes)" % (os.path.basename(track), len(notes)))
        out = a.out or os.path.join(
            HERE, "out", "circles_%s_%d.mp4" % (time.strftime("%Y%m%d_%H%M%S"), k + 1))
        pack = random.choice(packs.all_packs()) if a.pack == "random" else a.pack
        ids = packs.pick(pack, a.countries)
        print("pack: %s  -  %s" % (pack, ", ".join(ids)))
        fin = [c.strip().lower() for c in a.finalists.split(",") if c.strip()]
        if a.finalists and len(fin) != 2:
            sys.exit("--finalists needs exactly two ids, e.g. ua,us")
        if a.winner and fin and a.winner.lower() not in fin:
            sys.exit("--winner has to be one of --finalists")
        # A drawn roster need not contain who was asked for, so they are seated
        # into it - otherwise `--countries random8 --winner ua` is a coin flip
        # on whether the run can even be searched for.
        for c in dict.fromkeys([x for x in [a.winner.lower()] + fin if x]):
            if c not in ids:
                packs.image_path(pack, c)          # fetch its picture like pick did
                free = [k for k, x in enumerate(ids) if x not in fin and x != a.winner.lower()]
                ids[random.choice(free)] = c
                print("  seated %s into the draw" % c)
        want = ids.index(a.winner.lower()) if a.winner else None
        pair = {ids.index(c) for c in fin} if fin else None
        hook = random.choice(tags.hooks(pack)) if a.hook == "random" else a.hook
        print(one(pack, ids, hook, a.seed, out, a.preview, notes, want, pair))


if __name__ == "__main__":
    main()
