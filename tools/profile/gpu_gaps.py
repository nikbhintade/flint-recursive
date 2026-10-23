"""Summarize GPU activity in a decode window of an nsys sqlite export: busy share and idle gaps between kernels.

Usage: python3 tools/profile/gpu_gaps.py data/profile/<name>.sqlite [window_fraction=0.5]
The window is the last fraction of the run's kernel activity (skips model load and the prompt).
"""
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ev = db.execute("SELECT start, end, 'k' FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
ev += db.execute("SELECT start, end, 'm' FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
ev.sort()
t0, t1 = ev[0][0], max(e for _, e, _ in ev)
w0 = t1 - (t1 - t0) * frac
ev = [x for x in ev if x[0] >= w0]
busy, gaps, cur_s, cur_e = 0, [], None, None
for s, e, _ in ev:
    if cur_e is None or s > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
        cur_s, cur_e = s, e
    else:
        cur_e = max(cur_e, e)
busy += cur_e - cur_s
win = cur_e - ev[0][0]
big = [g for g in gaps if g > 2000]
print(f"window {win/1e6:.1f} ms: GPU busy {busy/1e6:.1f} ms ({busy/win:.0%}); "
      f"{len(big)} gaps > 2 us totalling {sum(big)/1e6:.1f} ms (median {sorted(big)[len(big)//2]/1e3:.1f} us)")
print(f"kernels {sum(1 for x in ev if x[2]=='k')}, memcpys {sum(1 for x in ev if x[2]=='m')}")
