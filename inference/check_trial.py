"""Trial-log gate for an rsna-blend4-* kernel. usage: check_b4.py <log> [extra marker ...]; prints 1 or 0, details on stderr."""
import json, sys
try:
    t = "\n".join(e.get("data", "") for e in json.load(open(sys.argv[1], encoding="utf-8")))
except Exception as e:
    print(0); print("unreadable", e, file=sys.stderr); sys.exit()
import os
need = [os.environ.get("CKPT_MARK", "10 checkpoints"), "0 studies median-filled", "OURS_STAGE rc 0 ok True", "MEMBER_DONE s6 scored 3",
        "MEMBER_DONE r7 scored 3", "BLEND4_WRITTEN"] + sys.argv[2:]
miss = [k for k in need if k not in t]
bad = "STAGE_FAILED" in t
print(int(not miss and not bad))
print("missing", miss, "stage_failed", bad, file=sys.stderr)
i = t.find("BLEND4_WRITTEN")
print(t[i:i + 420] if i >= 0 else t[-600:], file=sys.stderr)
