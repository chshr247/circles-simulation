"""Flag-circles 9:16 video generator.

Six rulesets over one arena, one camera and one soundtrack, picked with --mode:

  tether  balls spawn with 10-12 lines tethered to the arena wall. They fly and
          bounce (off each other and the wall); every bounce speeds them up. A
          ball crossing someone else's tether cuts it - fans cross each other
          freely, the cut is what matters. Touching the wall sticks a new
          tether on at the point of contact, and it stays there. No tethers left -> the ball dies. Last ball standing wins.
  escape  a travelling, widening opening in the rim; falling out costs a life
          of three, and the last one with any left wins.
  climb   a pachinko shaft of pegs and sliding bars, a red line closing in from
          behind, and a finish line laid ahead of the leader at 45 seconds.
  paint   they paint the floor and last place is culled off it every six
          seconds, down to a two-way final.
  bomb    hot potato: a lit fuse passed on contact, and whoever holds it when
          it burns down loses one of two lives.
  zone    king of the hill: seconds banked while inside a closing circle in the
          middle, double for whoever is in it alone, first to the target wins.

See MODES near the bottom of the simulation half for what a ruleset owes the
rest of the file.
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
DECAY_EVERY = 2.0            # sudden death: seconds between forced tether losses
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
# The window a run may land in, or find_seed throws the seed away. Was 88, and
# a minute and a half is a video that gets abandoned rather than finished -
# which is the number that decides whether it gets shown to anybody else.
#
# Two things this top is NOT. It is the length BEFORE finish() stretches the
# last seconds into the slow finish, and that rides ~7s on top of every ruleset
# but paint, so 50 here is ~57s of mp4. And it cannot go much under 50 anyway:
# tether, paint and bomb are time-driven and land at 46-49s whatever the seed,
# so a tighter top would not shorten them, it would leave find_seed nothing to
# match and kill those three outright. escape, climb and zone are the only ones
# it actually trims, and about half their seeds still fit - find_seed tries 400.
MIN_DUR, MAX_DUR = 30.0, 50.0
TRAIL_FRAMES, TRAIL_MIN = 16, 0.22   # paint's comet tail: length, and the speed
                                     # below which a ball does not have one
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
    __slots__ = ("i", "x", "y", "vx", "vy", "r", "lines", "alive", "spd", "ready",
                 "bounces", "next_bounce", "crack", "score")

    def __init__(self, i, x, y, vx, vy, lines):
        self.i, self.x, self.y, self.vx, self.vy = i, x, y, vx, vy
        self.r, self.lines, self.alive = BALL_R, lines, True
        self.spd = SPEED0                  # only ever grows - bounces never slow a ball
        self.ready = 0.0                   # next time this ball may cut again
        self.bounces = 0                   # any ricochet - the only thing that speeds a ball up
        self.next_bounce = 0.0             # see _ricochet
        self.crack = -1.0                  # -1 alive, 0..1 breaking apart
        self.score = 0.0                   # what the leaderboard ranks by, mode's own unit


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


def _steer(b, dx, dy, most):
    """Turn the ball at most `most` radians towards (dx, dy). Speed is not
    touched - steering is a heading, and anything that changed the speed here
    would fight the ruleset's own clock for it."""
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return
    cur = math.atan2(b.vy, b.vx)
    d = (math.atan2(dy, dx) - cur + math.pi) % math.tau - math.pi
    a = cur + max(-most, min(most, d))
    _aim(b, math.cos(a), math.sin(a))


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


def _bounce_pairs(alive, rnd, t, cap, events):
    """Ball on ball, elastic, separated first so nobody stays overlapped. Every
    ruleset that is not the falling shaft shares it - three copies of the same
    approach test is three places to get it wrong."""
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a, c = alive[i], alive[j]
            dx, dy = c.x - a.x, c.y - a.y
            d = math.hypot(dx, dy) or 1e-9
            if d >= a.r + c.r:
                continue
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


def space_ring(vals, sep, full=math.tau):
    """Push points apart around a circle until no two are closer than `sep`,
    each moving as little as it can. Returns them in the order given.

    Cut the ring at its widest empty stretch and it becomes a plain line to
    space out. Pushing pairs apart in a loop never converged - every push
    re-ordered the ring behind it.

    Anchors on the rim and the hues the balls are drawn in are the same problem
    at different scales, so they are the same routine: a set of points that each
    want to sit somewhere in particular and may not sit on top of each other.
    """
    n = len(vals)
    if n < 2:
        return list(vals)
    order = sorted(range(n), key=lambda k: vals[k] % full)
    ring = [vals[k] % full for k in order]
    cut, widest = 0, -1.0
    for i in range(n):
        d = (ring[i + 1] - ring[i]) if i + 1 < n else (ring[0] + full - ring[i])
        if d > widest:
            widest, cut = d, (i + 1) % n
    base = ring[cut]
    pos = [(ring[(cut + k) % n] - base) % full for k in range(n)]
    for k in range(1, n):
        pos[k] = max(pos[k], pos[k - 1] + sep)
    if pos[-1] > full - sep:                               # overflowed the cut
        pos[-1] = full - sep
        for k in range(n - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - sep)
    out = [0.0] * n
    for k in range(n):
        out[order[(cut + k) % n]] = (base + pos[k]) % full
    return out


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
            _bounce_pairs(alive, rnd, t, cap, events)          # ball vs ball
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
        if t > REWARD_ZERO and t > next_decay:
            # Two balls can genuinely never meet, and the run would sit there
            # forever. Past the window everyone starts shedding tethers whether
            # or not anyone cuts them.
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
            frames.append([(b.x, b.y, b.i, tuple(b.lines), b.crack, len(b.lines))
                           for b in balls if b.alive])
        if ending is not None and t - ending >= DEATH_MOVE + CRACK_SECS:
            doomed.alive = False
            events.append(("gone", t))
            if record:                       # one clean frame without it, or the
                frames.append([(b.x, b.y, b.i, tuple(b.lines), b.crack, len(b.lines))
                               for b in balls if b.alive])   # hold freezes the corpse
        alive = [b for b in balls if b.alive]
        if duel_at is None and len(alive) <= 2:
            duel_at = t
            events.append(("duel", t))
        if len(alive) <= 1:
            return dict(frames=frames, events=events, duration=t, seed=seed,
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
    # A mode's backdrop - the travelling opening, the closing chase - moves on
    # the same clock as the balls, so it is stretched with them. Held instead of
    # interpolated it strobes once per five frames right where the eye is.
    ex = run.get("extra") or []
    ex_head, ex_tail = ex[:len(head)], ex[len(head):]

    out, out_ex = list(head), list(ex_head)
    for i in range(len(tail) - 1):
        a = {row[2]: row for row in tail[i]}
        b = {row[2]: row for row in tail[i + 1]}
        for st in range(steps):
            f = st / steps
            out.append([(a[k][0] + (b[k][0] - a[k][0]) * f,
                         a[k][1] + (b[k][1] - a[k][1]) * f,
                         k, b[k][3],
                         a[k][4] + (b[k][4] - a[k][4]) * f,
                         a[k][5] + (b[k][5] - a[k][5]) * f) for k in a if k in b])
            if ex_tail:
                out_ex.append(tuple(p + (q - p) * f
                                    for p, q in zip(ex_tail[i], ex_tail[i + 1])))
    out.append(tail[-1])
    if ex_tail:
        out_ex.append(ex_tail[-1])
    for _ in range(int(HOLD_S * FPS)):            # hold on the winner
        out.append(out[-1])
        if ex_tail:
            out_ex.append(out_ex[-1])
    if ex:
        run["extra"] = out_ex

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


def find_seed(seed_arg, n_balls, want=None, pair=None, limit=400, shortlist=8,
              sim=None):
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
    sim = sim or simulate
    if pair:
        limit, shortlist = limit * 10, 3
    start = random.randrange(1 << 30)
    best, tried = [], 0
    for k in range(limit):
        s = start + k
        tried = k + 1
        r = sim(s, n_balls, record=False)
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


# ---------------------------------------------------------------- other modes
# Three more rulesets over the same arena, camera, soundtrack and packs. A mode
# is only a simulate() filling the same run dict: frames of
# (x, y, i, lines, crack, score) rows, events the melody rides, and one tuple of
# floats per frame in `extra` for whatever sits behind the balls. Everything
# downstream - the seed search, the slow finish, the leaderboard, the caption -
# is already mode-blind, so a fourth ruleset is one function and one table row.

# escape: the rim has a single opening, it travels, and it widens. The opening
# is drawn wide because a hole a ball cannot be seen to fit through reads as a
# glitch - but the whole ball has to clear it, and a ball is 16 degrees of this
# rim, so the gap a run actually plays against is what is left after that.
GAP_DEG, GAP_GROW, GAP_SPIN = 28.0, 0.06, 18.0    # deg, deg/s, deg/s
LIVES = 3                    # hearts a ball starts with; falling out costs one
ESCAPE_SPEED = 700.0

# climb: a pachinko shaft the pack falls down, with the trailing edge closing in
SHAFT_X = 70                 # wall inset
GRAV, FALL_MAX = 2400.0, 1100.0
PEG_R, PEG_DY, PEG_COLS = 26, 250, 3
# A peg that ate a third of the speed turned the shaft into a lottery: threading
# the gaps meant free fall, catching two pegs meant a stall, and the field was
# 1500 px deep inside three seconds. Deflecting instead of absorbing is what
# lets the pegs shuffle the order without also setting the pace.
PEG_DAMP = 0.92              # share of the speed a peg leaves the ball
CATCH = 2.5                  # gravity swing across a frame-height of the field
CHASE0, CHASE_MIN, CHASE_RATE = 1800.0, 240.0, 14.0    # px, px, px/s
CLIMB_SCALE = 100.0          # px of shaft per point on the scoreboard
FINISH_AT, FINISH_GAP = 45.0, 2400.0   # when the line is laid, and how far ahead
# Every third row of pegs is a sliding bar instead. Pegs alone scatter the order
# but they scatter it evenly; a bar is the obstacle you can be on the wrong side
# of, and one that moves is the wrong side arriving on its own.
BAR_EVERY, BAR_SPAN, BAR_H = 3, 0.38, 22
BAR_AMP, BAR_HZ = 0.21, 0.20           # share of the shaft, cycles/s

# paint: the floor is the scoreboard, and last place is culled off it on a clock
PAINT_R, PAINT_SPEED, PAINT_GRID = 40, 560.0, 5
PAINT_SPEED_MAX = 1500.0     # they wind up on the same clock the others do
CULL_EVERY = 6.0             # seconds between cuts
PAINT_FINAL = 12.0           # ...and the head to head once two are left
BAR_Y = 548                  # the countdown to the next cut, just above the rim


def _spawn_free(rnd, n, speed):
    """The spawn ring without tethers - every ruleset but the first one."""
    balls = _spawn(rnd, n)
    for b in balls:
        b.lines = []
        b.spd = speed
        _aim(b, b.vx, b.vy)
    return balls


def _finish(balls, b, t, events):
    """A ball is out. Returns (ending, doomed) when that was the second to last:
    the kill is held on screen instead of blinking out - the same hold the first
    ruleset ends on, and the stretch the slow motion is cut for.

    Held or not, the ball leaves `crack` behind at 0, which is what marks it out
    of the running. Whether it then shatters is the caller's: `escape` does not
    grow it, because there the loser leaves through the opening and the arena is
    wider than the video - a ball a hundred pixels past the rim at three o'clock
    is already off the right edge, and breaking it up there is a payoff nobody
    sees."""
    events.append(("die", t))
    if sum(1 for x in balls if x.alive and x is not b) == 1:
        b.crack = 0.0
        return t, b
    b.alive = False
    return None, None


def sim_escape(seed, n_balls, record=True):
    """Falling out costs a life, not the run. Three each, and the scoreboard is
    the count.

    The first cut of this had one exit eliminate you outright, and it was the
    ruleset nobody could read: death was a coin flip on whether the hole
    happened to be where you hit the wall, the number on the board was near
    misses - trivia that ranked the loser above the winner - and there was
    nothing on screen that got worse for anybody until they were gone. Lives fix
    all three at once. The board now counts down, so a ball in trouble is
    visible as a ball in trouble; every exit is an event instead of a removal,
    which is roughly one every two seconds instead of one every six; and the
    opening can be drawn big enough to be the obvious point of the video,
    because falling through it is survivable."""
    rnd = random.Random(seed)
    balls = _spawn_free(rnd, n_balls, ESCAPE_SPEED)
    for b in balls:
        b.score = LIVES
    frames, events, extra = [], [], []
    t, esc_step = 0.0, 0
    ending, doomed, duel_at = None, None, None
    gap0 = rnd.uniform(0, math.tau)
    # Clearing the lip commits a ball: from there the rim is behind it and it
    # gets no more wall. Without this the opening, which travels at 30 deg/s,
    # slides back under a ball still on its way out - and the wall it has just
    # left snaps it from beyond the rim back to the inside face of it. Measured
    # on seed 368079979: the loser cracked at radius 391, well inside an arena
    # it had already left.
    gone_out = set()
    while t < HARD_STOP:
        vmax = max((b.spd for b in balls if b.alive), default=ESCAPE_SPEED)
        sub = max(SUB_MIN, min(SUB_MAX, int(vmax / FPS / STEP_PX) + 1))
        h = 1.0 / FPS / sub
        for _ in range(sub):
            alive = [b for b in balls if b.alive]
            cap = EARLY_SPEED_MAX if len(alive) > 2 else SPEED_MAX
            g = gap0 + math.radians(GAP_SPIN) * t
            half = math.radians(min(GAP_DEG + GAP_GROW * t, 110.0)) / 2
            roll = 1.0 if ending is None else max(0.0, 1.0 - (t - ending) / DEATH_MOVE)
            for b in alive:
                b.x += b.vx * h * roll
                b.y += b.vy * h * roll
                d = math.hypot(b.x - CX, b.y - CY) or 1e-9
                if b.i in gone_out:
                    if d - b.r > R and ending is None:
                        b.score -= 1
                        gone_out.discard(b.i)
                        if b.score > 0:              # back in at the middle, at
                            events.append(("die", t))     # a fresh angle
                            b.x, b.y = CX, CY
                            _aim(b, math.cos(a := rnd.uniform(0, math.tau)), math.sin(a))
                        else:
                            ending, doomed = _finish(balls, b, t, events)
                    continue
                if d + b.r <= R:
                    continue
                off = abs((math.atan2(b.y - CY, b.x - CX) - g + math.pi) % math.tau - math.pi)
                if off < half - math.asin(b.r / R):  # the whole ball is over it
                    if d > R:                        # ...and its centre is past
                        gone_out.add(b.i)            # the rim, so it is committed
                    continue                         # either way there is no wall
                                                     # here to bounce off
                nx, ny = (b.x - CX) / d, (b.y - CY) / d
                b.x, b.y = CX + nx * (R - b.r), CY + ny * (R - b.r)
                vn = b.vx * nx + b.vy * ny
                _ricochet(b, t, cap)
                _aim(b, *_jitter(rnd, b.vx - 2 * vn * nx, b.vy - 2 * vn * ny, -nx, -ny))
                events.append(("wall", t))
            if ending is None:
                _bounce_pairs(alive, rnd, t, cap, events)
            t += h
        live = [b for b in balls if b.alive]
        step_now = int(t / ESCALATE_EVERY)
        if step_now > esc_step:
            esc_step = step_now
            cap = EARLY_SPEED_MAX if len(live) > 2 else SPEED_MAX
            for b in live:
                b.spd = min(cap, b.spd * (1.0 + ESCALATE_PCT / 100.0))
        if record:
            frames.append([(b.x, b.y, b.i, (), b.crack, b.score)
                           for b in balls if b.alive])
            extra.append((gap0 + math.radians(GAP_SPIN) * t,
                          math.radians(min(GAP_DEG + GAP_GROW * t, 110.0))))
        if ending is not None and t - ending >= DEATH_MOVE + CRACK_SECS:
            doomed.alive = False
            events.append(("gone", t))
        if duel_at is None and sum(1 for b in balls if b.alive) <= 2:
            duel_at = t
            events.append(("duel", t))
        alive = [b for b in balls if b.alive]
        if len(alive) <= 1:
            return dict(frames=frames, events=events, extra=extra, duration=t,
                        seed=seed, winner=alive[0].i if alive else None,
                        runner=doomed.i if doomed else None,
                        duel=t - (duel_at if duel_at is not None else t))
    return None


def pegs_near(seed, top, bottom):
    """The shaft's pegs over a stretch of it, derived from the row number. The
    sim and the renderer each ask for the rows they need instead of the run
    carrying a peg field through every one of four thousand frames."""
    out = []
    span = W - 2 * SHAFT_X
    for row in range(max(2, int(top // PEG_DY)), int(bottom // PEG_DY) + 1):
        if row % BAR_EVERY == 0:
            continue                     # that row belongs to a bar
        rnd = random.Random(seed * 7919 + row)
        for c in range(PEG_COLS):
            x = SHAFT_X + span * (c + 0.5 + (0.5 if row % 2 else 0.0)) / PEG_COLS
            if x > W - SHAFT_X - PEG_R:
                continue
            out.append((x + rnd.uniform(-30, 30), row * PEG_DY + rnd.uniform(-40, 40)))
    return out


def bars_near(seed, top, bottom, t):
    """The sliding bars over a stretch of the shaft, as (left, right, y, vx).
    Same deal as the pegs - derived from the row so the sim and the renderer
    agree without the run carrying them - except these also depend on the clock,
    which is why `t` rides along in every frame's backdrop tuple."""
    out = []
    span = W - 2 * SHAFT_X
    half = span * BAR_SPAN / 2
    for row in range(max(2, int(top // PEG_DY)), int(bottom // PEG_DY) + 1):
        if row % BAR_EVERY:
            continue
        ph = random.Random(seed * 7919 + row).uniform(0, math.tau)
        w = math.tau * BAR_HZ
        cx = W / 2 + span * BAR_AMP * math.sin(ph + w * t)
        out.append((cx - half, cx + half, float(row * PEG_DY),
                    span * BAR_AMP * w * math.cos(ph + w * t)))
    return out


def sim_climb(seed, n_balls, record=True):
    """A pachinko race down a shaft with no bottom. Gravity does all of it -
    the pegs only scatter who is in front, and they re-scatter it every couple
    of seconds, which is the point: a lead here never survives a peg field.

    Two things keep it from being over in ten seconds. Whoever is behind falls
    harder, so the pack stays one screen deep and the order keeps swapping; and
    the trailing edge closes in on the leader on a clock, so there is always an
    ending even when nobody would otherwise have lost.
    """
    rnd = random.Random(seed)
    balls, span = [], W - 2 * SHAFT_X - 2 * BALL_R
    for i in range(n_balls):
        col = i % max(1, n_balls // 2)
        wide = max(1, n_balls // 2)
        balls.append(Ball(i, SHAFT_X + BALL_R + span * (col + 0.5) / wide,
                          140 + (i // wide) * 2.4 * BALL_R + rnd.uniform(-20, 20),
                          rnd.uniform(-80, 80), 0.0, []))
    y0 = min(b.y for b in balls)
    frames, events, extra = [], [], []
    t, cam = 0.0, 0.0
    ending, doomed, cracked, duel_at = None, None, False, None
    finish, won = None, None
    while t < HARD_STOP:
        vmax = max((abs(b.vy) + abs(b.vx) for b in balls if b.alive), default=GRAV)
        sub = max(SUB_MIN, min(SUB_MAX, int(vmax / FPS / STEP_PX) + 1))
        h = 1.0 / FPS / sub
        pegs = pegs_near(seed, cam - PEG_DY, cam + H + PEG_DY)
        bars = bars_near(seed, cam - PEG_DY, cam + H + PEG_DY, t)
        for _ in range(sub):
            alive = [b for b in balls if b.alive]
            mid = sum(b.y for b in alive) / len(alive)
            roll = 1.0 if ending is None else max(0.0, 1.0 - (t - ending) / DEATH_MOVE)
            for b in alive:
                # Rubber band about the middle of the field: behind it a ball is
                # heavier, in front of it lighter. Measured without this the pack
                # strings out inside fifteen seconds and the rest of the video is
                # one ball alone on screen - and pulling only the stragglers is
                # not enough, because it is the runaway that sets the pace.
                b.vy += GRAV * min(1.0 + CATCH, max(0.35,
                                   1.0 + CATCH * (mid - b.y) / H)) * h
                b.vy = max(-FALL_MAX, min(FALL_MAX, b.vy))
                b.x += b.vx * h * roll
                b.y += b.vy * h * roll
                if b.x - b.r < SHAFT_X or b.x + b.r > W - SHAFT_X:
                    b.vx = -b.vx * PEG_DAMP
                    events.append(("wall", t))
                for px, py in pegs:
                    dx, dy = b.x - px, b.y - py
                    d = math.hypot(dx, dy) or 1e-9
                    if d >= b.r + PEG_R:
                        continue
                    nx, ny = dx / d, dy / d
                    b.x, b.y = px + nx * (b.r + PEG_R), py + ny * (b.r + PEG_R)
                    vn = b.vx * nx + b.vy * ny
                    if vn < 0:
                        b.vx = (b.vx - 2 * vn * nx) * PEG_DAMP
                        b.vy = (b.vy - 2 * vn * ny) * PEG_DAMP
                        b.vx += rnd.uniform(-90, 90)     # no two runs down the
                        events.append(("hit", t))        # same peg are identical
                for x0, x1, by, bvx in bars:
                    # A bar is a capsule: the segment between its ends, thickened
                    # by half its height. Same contact maths as a peg, so a ball
                    # rounds the end of one instead of catching on a corner.
                    cx_ = min(max(b.x, x0), x1)
                    dx, dy = b.x - cx_, b.y - by
                    d = math.hypot(dx, dy) or 1e-9
                    if d >= b.r + BAR_H / 2:
                        continue
                    nx, ny = dx / d, dy / d
                    b.x, b.y = cx_ + nx * (b.r + BAR_H / 2), by + ny * (b.r + BAR_H / 2)
                    vn = b.vx * nx + b.vy * ny
                    if vn < 0:
                        b.vx = (b.vx - 2 * vn * nx) * PEG_DAMP + bvx * 0.45
                        b.vy = (b.vy - 2 * vn * ny) * PEG_DAMP
                        events.append(("hit", t))
            _climb_pairs(alive)
            for b in alive:      # last word on the walls: a peg or a neighbour
                b.x = min(max(b.x, SHAFT_X + b.r), W - SHAFT_X - b.r)   # shoves
            t += h                                                      # sideways
        alive = [b for b in balls if b.alive]
        lead = max(b.y for b in alive)
        # Attrition alone ends a race whenever it happens to end, and the video
        # is over the moment the second-to-last ball dies - no finish, nothing to
        # race AT. So the line is laid at FINISH_AT, a fixed distance ahead of
        # whoever is in front, and from then on the run is a race to it: first
        # ball across wins outright, however many are still alive.
        # ...or the moment attrition has got it down to two, whichever lands
        # first. Without the second half a run the chase thinned quickly was
        # over before FINISH_AT and never saw the line at all - one rendered at
        # 37.8s that way, decided entirely from behind.
        if finish is None and (t >= FINISH_AT or len(alive) <= 2):
            finish = lead + FINISH_GAP
            if duel_at is None:              # the backing belongs under this
                duel_at = t
                events.append(("duel", t))
        # Once the line is down the chase stops dead, at the position it had
        # reached. It has to: a red line that is still eating the field decides
        # the run before anybody reaches the finish - measured on the first cut
        # of this, a race whose line was laid at 45s ended at 49.8s with the
        # last opponent killed from behind and the line never crossed. Frozen,
        # the field pulls away from it and it leaves over the top of the frame.
        if finish is None:
            kill = lead - max(CHASE_MIN, CHASE0 - CHASE_RATE * t)
        # The leader rides 62% down the frame, except while the red line is
        # further back than that leaves room for - then the camera pulls back to
        # keep the line on screen, because a ball dying above the top edge is an
        # elimination nobody saw. Never scrolls back: the finish is stretched off
        # this track and a camera that reversed would stutter through it.
        cam = max(cam, lead - H * 0.62 if finish is not None
                  else min(lead - H * 0.62, kill - 40))
        if ending is None and finish is not None:
            over = [b for b in alive if b.y > finish]
            if over:
                won = max(over, key=lambda b: b.y)
                ending = t
                events.append(("crack", t))
        if ending is None and finish is None:
            # Never past two: at two the line goes down instead, so the chase
            # can thin the field but never decide it. Sliced rather than looped
            # because two balls can cross the line in the same frame, and the
            # second of those would be the run ending from behind again.
            for b in [b for b in alive if b.y < kill][:max(0, len(alive) - 2)]:
                _finish(balls, b, t, events)
        if doomed is not None:
            doomed.crack = min(1.0, (t - ending) / (DEATH_MOVE + CRACK_SECS))
            if not cracked:
                cracked = True
                events.append(("crack", ending))
        # Frozen the instant the run is decided. The balls coast for another
        # half second after the line is crossed, and whoever came second can
        # roll further in that time than the winner did - which put the wrong
        # name at the top of the board on the final frame: winner India on 122,
        # Brazil above it on 123, having crossed second.
        if ending is None:
            for b in balls:
                if b.alive:
                    b.score = (b.y - y0) / CLIMB_SCALE
        if record:
            frames.append([(b.x, b.y - cam, b.i, (), b.crack, b.score)
                           for b in balls if b.alive])
            extra.append((kill - cam, cam, t,
                          H * 4.0 if finish is None else finish - cam))
        if ending is not None and t - ending >= DEATH_MOVE + CRACK_SECS:
            if doomed is not None:
                doomed.alive = False
                events.append(("gone", t))
            if won is not None:
                rest = [b for b in balls if b.alive and b is not won]
                return dict(frames=frames, events=events, extra=extra, duration=t,
                            seed=seed, winner=won.i,
                            runner=max(rest, key=lambda b: b.y).i if rest else None,
                            duel=t - (duel_at if duel_at is not None else t))
        if duel_at is None and sum(1 for b in balls if b.alive) <= 2:
            duel_at = t
            events.append(("duel", t))
        alive = [b for b in balls if b.alive]
        if len(alive) <= 1:
            return dict(frames=frames, events=events, extra=extra, duration=t,
                        seed=seed, winner=alive[0].i if alive else None,
                        runner=doomed.i if doomed else None,
                        duel=t - (duel_at if duel_at is not None else t))
    return None


def _climb_pairs(alive):
    """Ball on ball down the shaft. Its own copy of the collision because the
    shared one re-aims at a stored speed, and here speed is gravity's to set -
    normalising it would hand every clipped ball a free fall's worth of it."""
    for i in range(len(alive)):
        for j in range(i + 1, len(alive)):
            a, c = alive[i], alive[j]
            dx, dy = c.x - a.x, c.y - a.y
            d = math.hypot(dx, dy) or 1e-9
            if d >= a.r + c.r:
                continue
            nx, ny = dx / d, dy / d
            ov = (a.r + c.r - d) / 2
            a.x, a.y = a.x - nx * ov, a.y - ny * ov
            c.x, c.y = c.x + nx * ov, c.y + ny * ov
            van, vcn = a.vx * nx + a.vy * ny, c.vx * nx + c.vy * ny
            if van - vcn <= 0:
                continue
            a.vx += (vcn - van) * nx * PEG_DAMP
            a.vy += (vcn - van) * ny * PEG_DAMP
            c.vx += (van - vcn) * nx * PEG_DAMP
            c.vy += (van - vcn) * ny * PEG_DAMP


def sim_paint(seed, n_balls, record=True):
    """They paint the floor, and every CULL_EVERY seconds whoever owns the least
    of it is out and their colour is wiped off the ground. Two left, a twelve
    second head to head, most of the floor wins.

    The cut is the whole ruleset. Without it eight balls painting an arena that
    fills up in fifteen seconds all converge on a twelfth of it and stay there -
    measured 12.6, 12.5, 11.5, 11.5, 11.1, 10.5, 9.3, 7.8, a spread narrower
    than the noise, with no losing and nothing at stake. Culling last place
    turns the same physics into a number that has to keep climbing: every cut
    frees a dead ball's ground for whoever gets there first, and the survivors'
    shares roughly double each round on the way to a two-way split.

    The grid is the scoreboard, not the picture. It is a twenty-fifth of the
    frame's area, which is plenty to count percent with and cheap enough to
    tally on every one of three thousand frames; the renderer paints its own at
    full size off the same positions, and gets one mask per cut to wipe with.
    """
    rnd = random.Random(seed)
    balls = _spawn_free(rnd, n_balls, PAINT_SPEED)
    gw, gh = W // PAINT_GRID, H // PAINT_GRID
    grid = Image.new("L", (gw, gh), 0)
    gd = ImageDraw.Draw(grid)
    br = max(1, 2 * PAINT_R // PAINT_GRID)
    total = math.pi * (R / PAINT_GRID) ** 2
    frames, events, extra, culls = [], [], [], []
    t, duel_at, next_cull, span = 0.0, None, CULL_EVERY, CULL_EVERY
    base, esc_step = PAINT_SPEED, 0
    while t < HARD_STOP:
        alive = [b for b in balls if b.alive]
        # The whole field winds up together on the standing timer, and bounces
        # are held to that shared speed rather than banking their own. Paint
        # turns over faster and faster as the run goes, which is what stops the
        # last two rounds from being the same picture as the first two - and
        # keeping every ball on one number is what lets the ricochets stay
        # neutral, so a ball cannot bounce its way into outrunning the grid.
        step_now = int(t / ESCALATE_EVERY)
        if step_now > esc_step:
            esc_step = step_now
            base = min(PAINT_SPEED_MAX, base * (1.0 + ESCALATE_PCT / 100.0))
            for b in alive:
                b.spd = base
                _aim(b, b.vx, b.vy)
        sub = max(SUB_MIN, min(SUB_MAX, int(base / FPS / STEP_PX) + 1))
        h = 1.0 / FPS / sub
        was = [(b.x, b.y) for b in alive]
        for _ in range(sub):
            for b in alive:
                b.x += b.vx * h
                b.y += b.vy * h
                d = math.hypot(b.x - CX, b.y - CY) or 1e-9
                if d + b.r > R:
                    nx, ny = (b.x - CX) / d, (b.y - CY) / d
                    b.x, b.y = CX + nx * (R - b.r), CY + ny * (R - b.r)
                    vn = b.vx * nx + b.vy * ny
                    _aim(b, *_jitter(rnd, b.vx - 2 * vn * nx, b.vy - 2 * vn * ny,
                                     -nx, -ny))
                    events.append(("wall", t))
            _bounce_pairs(alive, rnd, t, base, events)
            t += h
        for b, (ox, oy) in zip(alive, was):      # one stroke a frame, not a substep
            gd.line([ox / PAINT_GRID, oy / PAINT_GRID, b.x / PAINT_GRID,
                     b.y / PAINT_GRID], fill=b.i + 1, width=br)
            gd.ellipse([(b.x - PAINT_R) / PAINT_GRID, (b.y - PAINT_R) / PAINT_GRID,
                        (b.x + PAINT_R) / PAINT_GRID, (b.y + PAINT_R) / PAINT_GRID],
                       fill=b.i + 1)
        cnt = np.bincount(np.asarray(grid).ravel(), minlength=n_balls + 1)
        for b in alive:
            b.score = 100.0 * cnt[b.i + 1] / total
        if t >= next_cull and len(alive) > 2:
            next_cull = t + CULL_EVERY
            out = min(alive, key=lambda b: (b.score, -b.i))
            out.alive = False
            events.append(("die", t))
            # Its ground goes back on the market. One mask per cut - seven for a
            # whole run - is what the renderer needs to wipe the same shape off
            # its full-size floor, and it is cheaper to hand it over than to have
            # the renderer track ownership per pixel for three thousand frames.
            a = np.asarray(grid)
            culls.append((len(frames), Image.fromarray(
                np.where(a == out.i + 1, 255, 0).astype(np.uint8), "L")))
            grid = Image.fromarray(np.where(a == out.i + 1, 0, a))
            gd = ImageDraw.Draw(grid)
            alive = [b for b in balls if b.alive]
            if len(alive) == 2:
                duel_at, next_cull, span = t, t + PAINT_FINAL, PAINT_FINAL
                events.append(("duel", t))
        if record:
            frames.append([(b.x, b.y, b.i, (), b.crack, b.score) for b in alive])
            extra.append((max(0.0, min(1.0, (next_cull - t) / span)),))
        if duel_at is not None and t >= duel_at + PAINT_FINAL:
            break
    order = sorted((b for b in balls if b.alive), key=lambda b: -b.score)
    events.append(("crack", t))                  # the bell, and the only bang in it
    return dict(frames=frames, events=events, extra=extra, duration=t, seed=seed,
                culls=culls, winner=order[0].i, runner=order[1].i,
                duel=t - (duel_at if duel_at is not None else t))


def paint_finish(run):
    """Only the hold. No slow motion, because there is no kill to cut it for,
    and deliberately no push-in either: what the run was about is the floor, and
    a camera that closes on the winner crops away the thing it won."""
    frames = run["frames"]
    if not frames:
        return run
    frames.extend([frames[-1]] * int((HOLD_S + 1.2) * FPS))
    run["extra"] += [(0.0,)] * (len(frames) - len(run["extra"]))
    run["duration"] = len(frames) / FPS
    return run


# bomb: one lit fuse in the field, handed on by touch, and whoever is holding it
# when it burns down pays for it
BOMB_SPEED = 620.0
FUSE0, FUSE_SHRINK, FUSE_MIN = 9.0, 0.90, 3.5   # seconds, and how each one shortens
# One touch hands it over, so the lock is only long enough to leave the contact:
# a pair stays overlapped for several substeps and without it the same collision
# passes the bomb, passes it back, and passes it again inside three frames.
PASS_LOCK = 0.10             # seconds after a pass in which the bomb cannot move
BOMB_CHASE = 1.34            # the holder's speed, against everyone else's
BOMB_TURN, FLEE_TURN = 2.6, 1.6   # rad/s the holder steers in, the field steers out
BLAST_R = 380.0              # how far a blast shoves whoever is still standing
BLAST_SECS = 0.45            # ...and how long its flash stays on screen

# zone: the middle is worth seconds and it closes as the run goes
ZONE_SPEED = 640.0
ZONE_R0, ZONE_R1, ZONE_CLOSE = 180.0, 110.0, 45.0    # px, px, seconds to close
ZONE_TARGET = 16.0           # seconds inside the circle to win it
ZONE_DRAG = 0.42             # share of its speed a ball keeps while it is in there
ZONE_SOLO = 2.0              # the clock runs this much faster for a ball in alone
ZONE_HOT = 0.72              # share of the target that brings the backing up
ZONE_TIGHT = 10.0            # percent between the top two that counts as a finish


def sim_bomb(seed, n_balls, record=True):
    """Hot potato. One ball carries a lit bomb, touching anybody hands it over,
    and the fuse does not care who is holding it when it ends: that ball is out
    of the run, and every fuse after it is shorter than the last.

    The steering is the ruleset and not a flourish. Left to plain billiards the
    bomb changes hands on whatever contact chance provides - a handful over a
    whole fuse in an arena this size - and half the blasts land on a ball that
    never had anyone within reach to pass to, which reads as a lottery. A holder
    that turns towards its nearest and a field that leans away from the holder
    turns every fuse into a chase with a visible target, and that chase is the
    only thing on screen that has to be read: whoever the bomb is pointed at is
    the one in trouble.

    Nothing winds up here - no per-ricochet speed, no escalation clock. The
    shortening fuse is the escalation, and it is the only one that reads: a
    field that is also flying faster every three seconds just turns the last
    two blasts into a scramble nobody can follow.

    The board counts saves - the times a ball had the bomb and got rid of it in
    time. With one life the obvious number is lives left, and that is a column
    of ones until it is a column of nothing; saves is the number that actually
    moves, and it says the true thing about a ball that keeps being handed the
    bomb and keeps surviving it.
    """
    rnd = random.Random(seed)
    balls = _spawn_free(rnd, n_balls, BOMB_SPEED)
    frames, events, extra = [], [], []
    t, base = 0.0, BOMB_SPEED
    ending, doomed, duel_at = None, None, None
    holder, giver = rnd.randrange(n_balls), -1
    fuse_len = FUSE0
    fuse_at, locked = FUSE0, 0.0
    blast = (CX, CY, -9.9)
    while t < HARD_STOP:
        alive = [b for b in balls if b.alive]
        for b in alive:                          # the one holding it is the fast one
            want = base * (BOMB_CHASE if b.i == holder else 1.0)
            if abs(b.spd - want) > 1e-6:
                b.spd = want
                _aim(b, b.vx, b.vy)
        cap = base * BOMB_CHASE
        sub = max(SUB_MIN, min(SUB_MAX, int(cap / FPS / STEP_PX) + 1))
        h = 1.0 / FPS / sub
        for _ in range(sub):
            hb = next((b for b in alive if b.i == holder), None)
            roll = 1.0 if ending is None else max(0.0, 1.0 - (t - ending) / DEATH_MOVE)
            if hb is not None and ending is None and len(alive) > 1:
                near = min((b for b in alive if b is not hb),
                           key=lambda b: (b.x - hb.x) ** 2 + (b.y - hb.y) ** 2)
                _steer(hb, near.x - hb.x, near.y - hb.y, BOMB_TURN * h)
                for b in alive:
                    if b is not hb:
                        _steer(b, b.x - hb.x, b.y - hb.y, FLEE_TURN * h)
            for b in alive:
                b.x += b.vx * h * roll
                b.y += b.vy * h * roll
                d = math.hypot(b.x - CX, b.y - CY) or 1e-9
                if d + b.r > R:
                    nx, ny = (b.x - CX) / d, (b.y - CY) / d
                    b.x, b.y = CX + nx * (R - b.r), CY + ny * (R - b.r)
                    vn = b.vx * nx + b.vy * ny
                    _aim(b, *_jitter(rnd, b.vx - 2 * vn * nx, b.vy - 2 * vn * ny,
                                     -nx, -ny))
                    events.append(("wall", t))
            if ending is None:
                _bounce_pairs(alive, rnd, t, cap, events)
            t += h
            if hb is None or ending is not None:
                continue
            if t >= locked:                      # touching the holder takes it
                # ...but not straight back to whoever just handed it over,
                # while there is anyone else to take it. Two balls that collide
                # are still touching on the next substep, so without this the
                # bomb spends the whole fuse rattling between the same pair
                # instead of travelling.
                back = len(alive) <= 2
                for b in alive:
                    if b is hb or not (back or b.i != giver):
                        continue
                    if math.hypot(b.x - hb.x, b.y - hb.y) <= b.r + hb.r + 2:
                        hb.score += 1            # got rid of it in time
                        holder, giver, locked = b.i, hb.i, t + PASS_LOCK
                        break
            if t < fuse_at:
                continue
            events.append(("crack", t))          # the bang
            blast = (hb.x, hb.y, t)
            for b in alive:
                if b is not hb and math.hypot(b.x - hb.x, b.y - hb.y) < BLAST_R:
                    _aim(b, b.x - hb.x, b.y - hb.y)
            fuse_len = max(FUSE_MIN, fuse_len * FUSE_SHRINK)
            fuse_at, locked = t + fuse_len, t + PASS_LOCK
            left = [b for b in alive if b is not hb]
            ending, doomed = _finish(balls, hb, t, events)
            if left and ending is None:          # it lands on whoever was nearest
                holder = min(left, key=lambda b: (b.x - hb.x) ** 2
                             + (b.y - hb.y) ** 2).i
                giver = -1
            # ...but a blast that ends the run does not hand anything on. The
            # winner spent the last stretch of the video holding a lit bomb with
            # a full fuse, which is the one thing the ruleset says you lose for.
            alive = [b for b in balls if b.alive]
        if ending is not None:
            doomed.crack = min(1.0, (t - ending) / (DEATH_MOVE + CRACK_SECS))
        if record:
            frames.append([(b.x, b.y, b.i, (), b.crack, b.score)
                           for b in balls if b.alive])
            extra.append((0.0 if ending is not None else
                          max(0.0, min(1.0, (fuse_at - t) / fuse_len)), float(holder),
                          max(0.0, 1.0 - (t - blast[2]) / BLAST_SECS),
                          blast[0], blast[1]))
        if ending is not None and t - ending >= DEATH_MOVE + CRACK_SECS:
            doomed.alive = False
            events.append(("gone", t))
        if duel_at is None and sum(1 for b in balls if b.alive) <= 2:
            duel_at = t
            events.append(("duel", t))
        alive = [b for b in balls if b.alive]
        if len(alive) <= 1:
            return dict(frames=frames, events=events, extra=extra, duration=t,
                        seed=seed, winner=alive[0].i if alive else None,
                        runner=doomed.i if doomed else None,
                        duel=t - (duel_at if duel_at is not None else t))
    return None


def sim_zone(seed, n_balls, record=True):
    """King of the hill. The middle of the arena is worth seconds, the rest of
    it is worth nothing, and the first ball to bank ZONE_TARGET of them takes
    the run. Nobody is ever out and what is banked stays banked, so every face
    on the board carries a number that only goes up - eight bars filling at once
    and a lead that changes hands on a bounce.

    Two knobs stop those eight bars filling at the same rate. A ball inside the
    circle is dragged down to ZONE_DRAG of its speed, so drifting in is sticky
    and one lucky entry is worth real seconds - and it makes whoever is scoring
    the slowest thing on the floor, which is to say the easiest to knock out of
    it. And a ball in there alone banks ZONE_SOLO times as fast, so a crowd in
    the middle is worth less to everyone in it than an empty circle is to one.

    The circle closes to ZONE_R1 over ZONE_CLOSE seconds. Early it is wide
    enough that everybody scores and the order keeps swapping; by the end it
    holds one ball, and the last seconds of the target have to be taken off
    whoever is already sitting in it.

    It ends on `slow_finish` like the rest and still gets no push-in, which is
    right and is not a special case: the camera backs off far enough to hold
    everybody who is still in the running, and here that is all eight to the
    last frame. So the slow motion lands on the bar filling and the frame keeps
    the seven bars it filled ahead of.
    """
    rnd = random.Random(seed)
    balls = _spawn_free(rnd, n_balls, ZONE_SPEED)
    held = [0.0] * n_balls
    frames, events, extra = [], [], []
    t, close_since, hot_at, solo = 0.0, 0.0, None, -1
    sub = max(SUB_MIN, min(SUB_MAX, int(ZONE_SPEED / FPS / STEP_PX) + 1))
    h = 1.0 / FPS / sub
    while t < HARD_STOP:
        zr = ZONE_R0 + (ZONE_R1 - ZONE_R0) * min(1.0, t / ZONE_CLOSE)
        for _ in range(sub):
            inside = [b for b in balls if math.hypot(b.x - CX, b.y - CY) < zr]
            rate = ZONE_SOLO if len(inside) == 1 else 1.0
            for b in inside:
                held[b.i] += h * rate
            ins = {b.i for b in inside}
            for b in balls:
                want = ZONE_SPEED * (ZONE_DRAG if b.i in ins else 1.0)
                if abs(b.spd - want) > 1e-6:
                    b.spd = want
                    _aim(b, b.vx, b.vy)
                b.x += b.vx * h
                b.y += b.vy * h
                d = math.hypot(b.x - CX, b.y - CY) or 1e-9
                if d + b.r > R:
                    nx, ny = (b.x - CX) / d, (b.y - CY) / d
                    b.x, b.y = CX + nx * (R - b.r), CY + ny * (R - b.r)
                    vn = b.vx * nx + b.vy * ny
                    _aim(b, *_jitter(rnd, b.vx - 2 * vn * nx, b.vy - 2 * vn * ny,
                                     -nx, -ny))
                    events.append(("wall", t))
            _bounce_pairs(balls, rnd, t, ZONE_SPEED, events)
            t += h
            solo = inside[0].i if len(inside) == 1 else -1
        for b in balls:
            b.score = 100.0 * held[b.i] / ZONE_TARGET
        order = sorted(balls, key=lambda b: (-b.score, b.i))
        if order[0].score - order[1].score > ZONE_TIGHT:
            close_since = t                      # the photo finish starts over
        if hot_at is None and order[0].score >= 100.0 * ZONE_HOT:
            hot_at = t
            events.append(("duel", t))           # the backing, not a head to head
        if record:
            frames.append([(b.x, b.y, b.i, (), b.crack, b.score) for b in balls])
            extra.append((zr, float(solo)))
        if order[0].score >= 100.0:
            events.append(("crack", t))          # the bell
            return dict(frames=frames, events=events, extra=extra, duration=t,
                        seed=seed, winner=order[0].i, runner=order[1].i,
                        duel=t - close_since)
    return None


MODES = {
    # sim, how the leaderboard prints a score, what turns the run into a video
    "tether": (simulate,   "%d",     slow_finish),
    "escape": (sim_escape, "%d",     slow_finish),
    "climb":  (sim_climb,  "%.0f",   slow_finish),
    "paint":  (sim_paint,  "%.0f%%", paint_finish),
    "bomb":   (sim_bomb,   "%d",     slow_finish),
    "zone":   (sim_zone,   "%.0f%%", slow_finish),
}


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


def synth(events, dur, notes=None, sr=44100, plain=False):
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
    if plain:
        lead = None      # --sfx plain: every event keeps its own noise, no tune
                         # is played over them and nothing is snapped to a grid
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
    if duel_t is not None and not plain:
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


def spread_hues(cols, min_gap=56.0):
    """Half the world's flags are red-dominant, so several balls come out the
    same colour. Force a minimum gap between every pair of hues around the
    wheel, each moving off its flag's own hue as little as that allows.

    The gap is deliberately wide enough that eight balls end up close to evenly
    spread. Fanning only the crowded clusters, which is what this did first,
    kept every ball nearer its flag - and still left Italy and Brazil as two
    greens a viewer had to compare side by side to tell apart, which in `paint`
    is not a detail: there the colour IS the territory. What survives the wider
    spread is the order, so the reddest flag still holds the reddest slot.
    """
    hsv = [list(colorsys.rgb_to_hsv(*(v / 255.0 for v in c))) for c in cols]
    raw = [h[0] * 360.0 for h in hsv]
    hues = space_ring(raw, min(min_gap, 360.0 / len(cols)), full=360.0)
    # Spacing anchors the ring at its widest gap and pushes everything forward
    # from there, so the whole set ends up rotated one way - Italy's green came
    # out 163 degrees round. Turning the finished ring back by the average of
    # those shifts costs nothing (a rotation cannot close a gap) and leaves
    # every ball about half as far from its own flag.
    off = sum(((h - r + 180.0) % 360.0) - 180.0 for h, r in zip(hues, raw)) / len(hues)
    hues = [(h - off) % 360.0 for h in hues]
    out = []
    for h, (_hue, s, _v) in zip(hues, hsv):
        r, g, b = colorsys.hsv_to_rgb(h / 360.0, max(s, 0.55), 1.0)
        out.append(tuple(int(round(x * 255)) for x in (r, g, b)))
    return out


def frame_heat(frames, mode, extra):
    """How fast each ball is on each frame, 0..1, straight off the frames.

    Derived rather than recorded: two consecutive positions already say it, and
    a number carried through the run would have to be interpolated by the slow
    finish like everything else. Deriving it means slow motion cools the colours
    on its own, which is right - during the stretch, they are moving slowly.

    In `climb` the stored y is a screen position and the camera chases the
    leader, so plain frame-to-frame motion says the leader is standing still and
    the stragglers are the fast ones. The camera offset rides in the backdrop
    tuple for exactly this kind of reason, so it goes back on first.

    The reference speed is a high percentile, not the maximum: `escape` puts a
    ball back at the middle of the arena when it loses a life, and that one
    teleport of five hundred pixels in a frame would otherwise define "fast" for
    the whole video and leave every real bounce looking cold.
    """
    out, seen = [], []
    for k, snap in enumerate(frames):
        off = extra[k][1] if mode == "climb" and k < len(extra) and extra[k] else 0.0
        prev = frames[k - 1] if k else []
        poff = extra[k - 1][1] if mode == "climb" and k and extra[k - 1] else 0.0
        was = {r[2]: (r[0], r[1] + poff) for r in prev}
        row = {}
        for r in snap:
            if r[2] in was:
                ox, oy = was[r[2]]
                row[r[2]] = math.hypot(r[0] - ox, r[1] + off - oy) * FPS
                seen.append(row[r[2]])
            else:
                row[r[2]] = 0.0
        out.append(row)
    seen.sort()
    ref = seen[int(len(seen) * 0.95)] if seen else 1.0
    return [{i: min(1.0, v / (ref or 1.0)) for i, v in row.items()} for row in out]


def hot(col, f, to=0.78):
    """The ball's colour at speed: the faster it is going, the further its
    outline is washed towards white. The disc under it is a flag and stays put -
    this is the ring and whatever the ring is dragging."""
    return tuple(int(round(c + (255 - c) * to * f)) for c in col)


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


def _bomb_art(dr, x, y, fuse, k):
    """The bomb itself, sat on the holder's shoulder so the face underneath it
    stays readable. Drawn rather than pasted: it is a handful of primitives
    against an asset to ship and scale, and the only thing it really has to do
    is get shorter. The fuse IS the clock - the stub burns down to nothing over
    the count, and the spark goes from yellow and small to red and wild.
    """
    bx, by, r = x + 58, y - 58, 44
    dr.ellipse([(bx - r) * SS, (by - r) * SS, (bx + r) * SS, (by + r) * SS],
               fill=(16, 16, 20), outline=(136, 136, 152), width=4 * SS)
    dr.ellipse([(bx - r * 0.52) * SS, (by - r * 0.74) * SS,
                (bx - r * 0.04) * SS, (by - r * 0.26) * SS], fill=(74, 74, 88))
    cx0, cy0 = bx + r * 0.36, by - r * 0.92          # the cap the fuse comes out of
    dr.line([(bx + r * 0.06) * SS, (by - r * 0.88) * SS, cx0 * SS, cy0 * SS],
            fill=(136, 136, 152), width=12 * SS)
    L = 12 + 44 * fuse
    pts = [(cx0, cy0)]
    for j in range(1, 5):
        f = j / 4.0
        pts.append((cx0 + L * f * 0.5 + 7 * math.sin(f * 3.2), cy0 - L * f))
    for j in range(len(pts) - 1):
        dr.line([pts[j][0] * SS, pts[j][1] * SS, pts[j + 1][0] * SS,
                 pts[j + 1][1] * SS], fill=(198, 182, 146), width=5 * SS)
    tx, ty = pts[-1]
    burn = 1.0 - fuse                                # hotter as it runs out
    col = (255, int(228 - 140 * burn), int(96 - 66 * burn))
    pu = 0.72 + 0.28 * math.sin(k * 1.7)
    for a in range(8):
        ang = a * math.pi / 4 + k * 0.25
        rr = (14 + 12 * burn) * pu
        dr.line([tx * SS, ty * SS, (tx + rr * math.cos(ang)) * SS,
                 (ty + rr * math.sin(ang)) * SS], fill=col, width=3 * SS)
    rr = (8 + 6 * burn) * pu
    dr.ellipse([(tx - rr) * SS, (ty - rr) * SS, (tx + rr) * SS, (ty + rr) * SS],
               fill=(255, 246, 214))


def draw_stats(img, snap, names, cols, f, fmt="%d"):
    """Live top-N by whatever the mode scores, under the arena. Drawn on the
    cropped frame rather than the arena layer, same as the hook: the finish
    zooms in, and a scoreboard that zooms with it walks straight off the bottom
    of the video."""
    d = ImageDraw.Draw(img)
    # A ball that is out is off the board the moment it goes, even while it is
    # still on screen falling or breaking up. It matters in `escape`, where the
    # score is near misses and the one that just fell out can be holding the
    # most of them: the video was ending on the loser's name above the winner's.
    rows = sorted((r for r in snap if r[4] < 0), key=lambda r: (-r[5], r[2]))[:STATS_TOP]
    for k, row in enumerate(rows):
        i, n = row[2], fmt % row[5]
        y = STATS_Y + k * STATS_ROW + STATS_ROW // 2
        d.rectangle([60, y - 13, 78, y + 13], fill=cols[i])
        name = names[i]
        while name and d.textlength(name, font=f) > 830:      # long ones exist
            name = name[:-1]
        d.text((98, y), name, font=f, fill=(255, 255, 255), anchor="lm",
               stroke_width=3, stroke_fill=(0, 0, 0))
        d.text((1020, y), n, font=f, fill=cols[i], anchor="rm",
               stroke_width=3, stroke_fill=(0, 0, 0))


def render(run, pack, ids, hook, out, preview=0, notes=None, mode="tether",
           plain=False):
    flags = [item_image(pack, i) for i in ids]
    roster = packs.roster(pack)
    names = [packs.display(roster.get(i, i)).upper() for i in ids]
    stats_font = font(38)
    fmt = MODES[mode][1]
    cols = spread_hues([dominant(f) for f in flags])
    discs = [disc(f, BALL_R * 2 * SS) for f in flags]

    base = Image.new("RGB", (W * SS, H * SS), (5, 5, 8))
    d = ImageDraw.Draw(base)
    rim = [(CX - R) * SS, (CY - R) * SS, (CX + R) * SS, (CY + R) * SS]
    if mode in ("tether", "paint", "bomb", "zone"):
        d.ellipse(rim, outline=(255, 255, 255), width=3 * SS)
    elif mode == "climb":                        # the shaft's two walls
        d.line([SHAFT_X * SS, 0, SHAFT_X * SS, H * SS], fill=(255, 255, 255), width=3 * SS)
        d.line([(W - SHAFT_X) * SS, 0, (W - SHAFT_X) * SS, H * SS],
               fill=(255, 255, 255), width=3 * SS)
    # Paint lands on a layer that keeps it: every frame copies the floor as it
    # stands and draws the balls over it, which is one line and one blob a ball
    # a frame instead of replaying the whole run's brush strokes each time.
    canvas = base.copy() if mode == "paint" else base
    canvas_d = ImageDraw.Draw(canvas)

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
    if tally and not plain:
        ru = {"cut": "срез", "wall": "стенка", "hit": "столкновение"}
        order = sorted(tally.items(), key=lambda kv: -kv[1])
        print("  мелодия на: %s   (%s)" % (ru[order[0][0]],
              ", ".join("%s %d" % (ru[k], v) for k, v in order)))
    write_wav(wav, synth(heard, dur, notes, plain=plain))

    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
           "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    extra = run.get("extra") or []
    culls = dict(run.get("culls") or [])
    carrier, owner = -1, -1
    heat = frame_heat(frames, mode, extra)
    tails = {}
    for k, snap in enumerate(frames):
        ex = extra[k] if k < len(extra) else ()
        hot_f = heat[k]
        if k in culls:
            # The culled ball's ground, wiped back to bare floor. The mask comes
            # off the sim's coarse grid, so its edge is blocky at this size -
            # which reads as the colour crumbling away, and is one paste per cut
            # against tracking ownership per pixel all run.
            m = culls[k].resize((W * SS, H * SS), Image.NEAREST)
            canvas.paste((5, 5, 8), (0, 0), m)
        if mode == "paint" and k:
            prev = {r[2]: r for r in frames[k - 1]}
            for x, y, i, _l, _c, _s in snap:
                if i in prev:
                    canvas_d.line([prev[i][0] * SS, prev[i][1] * SS, x * SS, y * SS],
                                  fill=cols[i], width=2 * PAINT_R * SS)
                canvas_d.ellipse([(x - PAINT_R) * SS, (y - PAINT_R) * SS,
                                  (x + PAINT_R) * SS, (y + PAINT_R) * SS], fill=cols[i])
        img = canvas.copy()
        dr = ImageDraw.Draw(img)
        if mode == "paint":
            dr.ellipse(rim, outline=(255, 255, 255), width=3 * SS)   # paint buried it
            if ex:      # how long last place has left, draining above the rim
                dr.rectangle([60 * SS, BAR_Y * SS, 1020 * SS, (BAR_Y + 12) * SS],
                             fill=(40, 40, 48))
                dr.rectangle([60 * SS, BAR_Y * SS, (60 + 960 * ex[0]) * SS,
                              (BAR_Y + 12) * SS],
                             fill=(235, 40, 60) if ex[0] < 0.3 else (235, 235, 245))
        elif mode == "escape" and ex:
            # The rim minus the opening: PIL sweeps an arc clockwise, so start
            # at one lip of the gap and end at the other the long way round.
            dr.arc(rim, math.degrees(ex[0] + ex[1] / 2),
                   math.degrees(ex[0] - ex[1] / 2), fill=(255, 255, 255), width=3 * SS)
        elif mode == "bomb" and ex:
            # The fuse twice over: the bar above the rim is the one you read
            # without looking away from the chase, the ring on the holder is the
            # one that says whose problem it is.
            dr.rectangle([60 * SS, BAR_Y * SS, 1020 * SS, (BAR_Y + 12) * SS],
                         fill=(40, 40, 48))
            dr.rectangle([60 * SS, BAR_Y * SS, (60 + 960 * ex[0]) * SS,
                          (BAR_Y + 12) * SS],
                         fill=(235, 40, 60) if ex[0] < 0.35 else (235, 235, 245))
            if ex[2] > 0.01:                     # the last blast, still expanding
                rr = (1.25 - ex[2]) * BLAST_R
                dr.ellipse([(ex[3] - rr) * SS, (ex[4] - rr) * SS,
                            (ex[3] + rr) * SS, (ex[4] + rr) * SS],
                           outline=(255, int(90 + 150 * ex[2]), 40),
                           width=max(1, int(4 + 20 * ex[2])) * SS)
        elif mode == "zone" and ex:
            # The circle takes the colour of whoever is in it alone, because
            # that is when it pays double - the tint IS the rule. Whose it is
            # rides in an interpolated channel, same as the bomb's carrier, so
            # anything between two whole numbers is the slow finish easing from
            # one owner to the next and the old one holds until it lands.
            # ...and it is drawn a ball's width out from the radius the sim
            # measures, because the sim measures centres. Drawn on the radius
            # itself, a ball whose centre is a pixel outside still overlaps most
            # of the circle - it is plainly standing in it and plainly not
            # scoring, which is the one thing a rule on screen may not do.
            # Out here the drawn edge means what it looks like it means: inside
            # is inside, and a ball that is not wholly in it is not in it.
            zr = ex[0] + BALL_R
            if abs(ex[1] - round(ex[1])) < 1e-6:
                owner = int(round(ex[1]))
            who = owner
            dr.ellipse([(CX - zr) * SS, (CY - zr) * SS,
                        (CX + zr) * SS, (CY + zr) * SS],
                       outline=cols[who] if 0 <= who < len(cols) else (86, 86, 100),
                       width=(9 if 0 <= who < len(cols) else 5) * SS)
        elif mode == "climb" and ex:
            top, bot = ex[1] - PEG_DY, ex[1] + H + PEG_DY
            for px, py in pegs_near(run["seed"], top, bot):
                sy = (py - ex[1]) * SS
                dr.ellipse([(px - PEG_R) * SS, sy - PEG_R * SS,
                            (px + PEG_R) * SS, sy + PEG_R * SS], fill=(120, 120, 132))
            for x0, x1, by, _vx in bars_near(run["seed"], top, bot, ex[2]):
                sy = (by - ex[1]) * SS
                dr.rounded_rectangle([x0 * SS, sy - BAR_H / 2 * SS,
                                      x1 * SS, sy + BAR_H / 2 * SS],
                                     radius=BAR_H / 2 * SS, fill=(150, 150, 164))
            if ex[3] < H + 60:                     # the line, once it is in shot
                sy = ex[3] * SS
                for c in range(0, W, 60):          # chequered, so it reads as one
                    dr.rectangle([c * SS, sy - 14 * SS, (c + 30) * SS, sy + 14 * SS],
                                 fill=(255, 255, 255))
                dr.rectangle([0, sy - 14 * SS, W * SS, sy + 14 * SS],
                             outline=(255, 235, 60), width=4 * SS)
            dr.rectangle([0, 0, W * SS, max(0.0, ex[0]) * SS], fill=(70, 8, 14))
            dr.line([0, ex[0] * SS, W * SS, ex[0] * SS], fill=(235, 40, 60), width=5 * SS)
        for x, y, i, lines, crack, _s in snap:
            for ang in lines:
                ax, ay = anchor_xy(ang)
                # Barely warmed, unlike the ring. A tether fan is how you tell
                # whose is whose and it covers most of the frame, so at duel
                # speed the full wash turned both fans white and the two balls
                # became one shape - measured on a 35 second final where the
                # only colour left on screen was the scoreboard's.
                dr.line([ax * SS, ay * SS, x * SS, y * SS],
                        fill=hot(cols[i], hot_f.get(i, 0.0), 0.3), width=3 * SS)
        if mode == "paint":
            # A comet tail, drawn on the frame and not on the floor, and only
            # once a ball is moving enough to have one. It has to be washed
            # towards white to be visible at all: it lies on top of the paint
            # that same ball has just laid down, and in its own flat colour it
            # would be invisible against it.
            for x, y, i, _l, _c, _s in snap:
                h = tails.setdefault(i, [])
                h.append((x, y))
                del h[:-TRAIL_FRAMES]
                f = hot_f.get(i, 0.0)
                if f < TRAIL_MIN or len(h) < 3:
                    continue
                span = h[-max(2, int(len(h) * f)):]
                for j in range(1, len(span)):
                    g = j / (len(span) - 1) * f
                    dr.line([span[j - 1][0] * SS, span[j - 1][1] * SS,
                             span[j][0] * SS, span[j][1] * SS],
                            fill=hot(cols[i], 0.35 + 0.6 * g),
                            width=max(1, int(BALL_R * 0.8 * g)) * SS)
        for x, y, i, lines, crack, _s in snap:
            px, py = int(x * SS) - BALL_R * SS, int(y * SS) - BALL_R * SS
            img.paste(discs[i], (px, py), discs[i])
            f = hot_f.get(i, 0.0)
            dr.ellipse([px, py, px + BALL_R * 2 * SS, py + BALL_R * 2 * SS],
                       outline=hot(cols[i], f), width=int(3 + 4 * f) * SS)
            if crack > 0:
                for seg in _crack_lines(x * SS, y * SS, BALL_R * SS, i, crack):
                    dr.line(seg, fill=(0, 0, 0), width=3 * SS)
        if mode == "bomb" and ex:
            # Who is holding it is an index, and the slow finish interpolates
            # every backdrop channel it is handed - so a hand-over inside the
            # stretch walks 6, 5, 4, 3, 2, 1 and spends four frames pointing at
            # balls that went out a minute ago. The last live one it named is
            # the answer for those frames: the ring sits still and then moves,
            # instead of blinking out at the one moment nobody blinks.
            hi = int(round(ex[1]))
            row = next((r for r in snap if r[2] == hi), None)
            if row is not None:
                carrier = hi
            else:
                row = next((r for r in snap if r[2] == carrier), None)
            if row and ex[0] > 0.002:            # a spent fuse leaves nothing
                _bomb_art(dr, row[0], row[1], ex[0], k)
        elif mode == "zone":
            # Everyone carries their own bar, so a viewer can follow one face
            # without reading the board under the arena.
            for x, y, i, _l, _c, s in snap:
                rr = (BALL_R + 15) * SS
                dr.arc([x * SS - rr, y * SS - rr, x * SS + rr, y * SS + rr],
                       -90, -90 + 3.599 * max(0.0, min(100.0, s)),
                       fill=cols[i], width=8 * SS)
        cx, cy, z = cam[k] if k < len(cam) else (W / 2.0, H / 2.0, 1.0)
        if z > 1.001:
            box = ((cx - W / (2 * z)) * SS, (cy - H / (2 * z)) * SS,
                   (cx + W / (2 * z)) * SS, (cy + H / (2 * z)) * SS)
            shot = img.resize((W, H), Image.LANCZOS, box=box)
        else:
            shot = img.resize((W, H), Image.LANCZOS)
        shot.paste(overlay, (0, 0), overlay)
        draw_stats(shot, snap, names, cols, stats_font, fmt)
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
    # anchors are stuck where they were struck: frame to frame the only angles
    # that are new are the ones a wall hit just added, never a drifted copy of
    # one that was already there
    prev = None
    for snap in simulate(4242, 8)["frames"]:
        now = {a for row in snap for a in row[3]}
        assert prev is None or len(now - prev) <= 16, "an anchor moved"
        prev = now
    b = Ball(0, 0, 0, 100.0, 0.0, [])        # steering is a heading, never a speed
    _steer(b, 0.0, 1.0, math.radians(30))
    assert abs(math.hypot(b.vx, b.vy) - b.spd) < 1e-9, "steering changed the speed"
    assert abs(math.degrees(math.atan2(b.vy, b.vx)) - 30) < 1e-6, "turned the long way"
    _steer(b, 0.0, 1.0, math.radians(90))    # ...and never past what it was given
    assert abs(math.degrees(math.atan2(b.vy, b.vx)) - 90) < 1e-6

    ok, durs = 0, []
    for s in range(12345, 12365):
        r = simulate(s, 8, record=False)
        if r:
            ok += 1
            durs.append(round(r["duration"], 1))
            assert r["winner"] is not None and 0 < r["duration"] <= HARD_STOP
            assert r["runner"] is not None and r["runner"] != r["winner"]
    print("selftest ok: %d/20 runs finished, durations %s" % (ok, sorted(durs)))
    # the leaderboard is sorted by the mode's own score, most first, ties by
    # index, and a ball that is out (crack >= 0) is off it however it scored
    snap = [(0, 0, 3, (), -1.0, 4), (0, 0, 1, (), -1.0, 9), (0, 0, 0, (), -1.0, 9),
            (0, 0, 2, (), 0.0, 99)]
    board = sorted((r for r in snap if r[4] < 0), key=lambda r: (-r[5], r[2]))
    assert [r[2] for r in board] == [0, 1, 3], board

    # hues end up at least the demanded gap apart, however crowded they started
    reds = [(200, 20, 20), (190, 30, 15), (210, 10, 40), (180, 40, 20), (255, 0, 0)]
    hs = sorted(colorsys.rgb_to_hsv(*(v / 255 for v in c))[0] * 360
                for c in spread_hues(reds))
    gaps = [hs[i + 1] - hs[i] for i in range(len(hs) - 1)] + [hs[0] + 360 - hs[-1]]
    want = min(56.0, 360.0 / len(reds))          # what spread_hues asks for
    assert min(gaps) > want - 1.0, sorted(round(g) for g in gaps)
    assert hot((0, 0, 0), 0.0) == (0, 0, 0) and hot((0, 0, 0), 1.0)[0] > 190

    # --sfx plain must leave the melody out: no snapping to the grid, no duel
    # backing, just the event's own noise where the event happened
    ev = [("cut", 0.30), ("cut", 0.31), ("duel", 0.5), ("die", 1.0)]
    import numpy as _np
    a_, b_ = synth(ev, 2.0), synth(ev, 2.0, plain=True)
    assert float(_np.abs(b_[int(44100 * 2.4):]).max()) < 1e-4, "plain kept a backing"
    assert float(_np.abs(a_[int(44100 * 2.4):]).max()) > 1e-3, "the backing went quiet"

    # speed comes off the frames, and climb's camera offset has to come back out
    # or the ball the camera is following reads as the slowest thing on screen
    fr = [[(0.0, 100.0, 0, (), -1.0, 0.0)], [(0.0, 100.0, 0, (), -1.0, 0.0)]]
    assert frame_heat(fr, "climb", [(0.0, 0.0), (0.0, 600.0)])[1][0] > 0.9
    assert frame_heat(fr, "paint", [(), ()])[1][0] == 0.0

    # Every ruleset has to finish, name a winner and a runner-up, and hand back
    # one backdrop tuple per frame - the slow finish interpolates the two lists
    # against each other and a short one would strobe the last second.
    for name, (sim, fmt, finish) in MODES.items():
        r = sim(4242, 8, record=True)
        assert r, "%s never finished inside HARD_STOP" % name
        assert r["winner"] is not None and r["runner"] != r["winner"], name
        ex = r.get("extra") or []
        assert not ex or len(ex) == len(r["frames"]), name
        assert all(len(row) == 6 for row in r["frames"][-1]), name
        fmt % r["frames"][-1][0][5]
        n = len(finish(r)["frames"])
        assert not (r.get("extra") or []) or len(r["extra"]) == n, name
        # One seed proves the ruleset runs; what the pipeline needs is that
        # find_seed does not have to hunt, so the window is checked over a
        # spread of seeds instead of pinned to whichever one is hard-coded here.
        durs = [x["duration"] for x in
                (sim(s, 8, record=False) for s in range(7100, 7108)) if x]
        good = sum(1 for x in durs if MIN_DUR <= x <= MAX_DUR)
        # Half of eight is not a hunt - find_seed tries 400 and takes the first
        # that fits. Below that the window has closed on the ruleset itself.
        assert good >= 4, (name, sorted(round(x) for x in durs))
        print("  %-6s seed 4242: %5.1fs, %d frames, winner #%d;  %d/8 seeds in "
              "%.0f-%.0fs" % (name, r["duration"], n, r["winner"], good,
                              MIN_DUR, MAX_DUR))
    _mode_checks()


def _mode_checks():
    """The two invariants the shared contract cannot see: a bomb that is being
    carried by a ball that is out of the run, and seconds banked in the middle
    going back down."""
    r = sim_bomb(4242, 8)
    for snap, ex in zip(r["frames"], r["extra"]):
        assert int(ex[1]) in {row[2] for row in snap}, "the bomb is on a dead ball"
    fuses = [ex[0] for ex in r["extra"]]
    assert min(fuses) >= 0.0 and max(fuses) <= 1.0, (min(fuses), max(fuses))

    r = sim_zone(4242, 8)
    was = {}
    for snap in r["frames"]:
        for row in snap:
            assert row[5] >= was.get(row[2], 0.0) - 1e-9, "banked seconds went down"
            was[row[2]] = row[5]
    assert max(was.values()) >= 100.0, "nobody ever filled it"
    print("  bomb/zone: the bomb stayed on a live ball, the middle only paid out")


def one(pack, ids, hook, seed_arg, out, preview, notes=None, want=None, pair=None,
        mode="tether", plain=False):
    sim, _fmt, finish = MODES[mode]
    # Nothing to shortlist when every run is the same length and the ending is
    # a tally rather than a duel - the first seed that answers is the answer.
    seed = find_seed(seed_arg, len(ids), want, pair,
                     shortlist=1 if mode == "paint" else 8, sim=sim)
    run = finish(sim(seed, len(ids)))
    winner = ids[run["winner"]]
    print("winner: %s  duration: %.1fs  events: %d"
          % (winner.upper(), run["duration"], len(run["events"])))
    render(run, pack, ids, hook, out, preview, notes, mode, plain)
    # Written beside the mp4 for whoever posts it. Nothing here can be worked
    # out from the file later: the countries are drawn at random and the winner
    # is only known once the run has been simulated.
    with open(os.path.splitext(out)[0] + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"hook": hook, "pack": pack, "mode": mode, "items": ids,
                   "winner": winner, "seed": seed,
                   "duration": round(run["duration"], 1),
                   "caption": tags.caption(hook, pack, ids, winner)}, f,
                  ensure_ascii=False, indent=1)
    return out


def main():
    global BALL_R
    ap = argparse.ArgumentParser()
    ap.add_argument("--hook", default="WHO WILL WIN",
                    help="text on the frame, or \"random\" for one per video")
    ap.add_argument("--pack", default=packs.DEFAULT,
                    help="what races: %s, or \"random\" for one per video"
                         % ", ".join(packs.all_packs()))
    ap.add_argument("--countries", default=DEFAULT_CC,
                    help="ids from the pack, or random8 / random10 to draw from it")
    ap.add_argument("--mode", default="tether",
                    help="ruleset: %s, or \"random\" for one per video"
                         % ", ".join(MODES))
    ap.add_argument("--seed", default="auto")
    ap.add_argument("--out", default="")
    ap.add_argument("--count", type=int, default=1, help="how many videos to render")
    ap.add_argument("--winner", default="", help="the id that must win")
    ap.add_argument("--finalists", default="",
                    help="the two ids that must be the last standing, e.g. ua,us")
    ap.add_argument("--preview", type=int, default=0, help="render only first N seconds")
    ap.add_argument("--melody", default=os.path.join(HERE, "sounds", "music"),
                    help="track, or a folder of them to draw from; \"\" for a plain scale")
    ap.add_argument("--sfx", default="melody", choices=("melody", "plain"),
                    help="melody: hits play the track's notes; plain: hits just "
                         "sound like hits")
    ap.add_argument("--trending", type=int, default=0,
                    help="pull N chart previews into the melody folder first")
    ap.add_argument("--trending-q", default="",
                    help="which chart, e.g. \"phonk\"; blank = global top")
    ap.add_argument("--ball-r", type=int, default=BALL_R,
                    help="ball radius in px; drop it when the field is crowded")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    BALL_R = a.ball_r
    if a.selftest:
        return selftest()
    plain = a.sfx == "plain"
    if a.trending and not plain and os.path.isdir(a.melody):
        print("trending: %d из %s" % (a.trending, a.trending_q or "мирового чарта"))
        fetch_trending(a.melody, a.trending, a.trending_q)
    pool = [] if plain else (melody_pool(a.melody) if a.melody else [])
    for k in range(a.count):
        notes = None
        if pool:                                   # a different track each video
            track = random.choice(pool)
            notes = melody_notes(track)
            print("melody: %s (%d notes)" % (os.path.basename(track), len(notes)))
        out = a.out or os.path.join(
            HERE, "out", "circles_%s_%d.mp4" % (time.strftime("%Y%m%d_%H%M%S"), k + 1))
        pack = random.choice(packs.all_packs()) if a.pack == "random" else a.pack
        mode = random.choice(list(MODES)) if a.mode == "random" else a.mode
        if mode not in MODES:
            sys.exit("unknown --mode %s; pick one of %s" % (mode, ", ".join(MODES)))
        ids = packs.pick(pack, a.countries)
        print("mode: %s  pack: %s  -  %s" % (mode, pack, ", ".join(ids)))
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
        hook = random.choice(tags.hooks(pack, mode)) if a.hook == "random" else a.hook
        print(one(pack, ids, hook, a.seed, out, a.preview, notes, want, pair, mode,
                  plain))


if __name__ == "__main__":
    main()
