"""Shadeform box for the GRAFT persona_qa / cities_qa reproduction (6 SmolLM2 runs on one GPU).

Uses the Shadeform helpers of unlearning-explorations (api(), managed_key(), ssh helpers); the API
key is read from ~/.shadeform/api_key. No server-side auto_delete; tear down explicitly.

  python box.py search      # read-only: offers for GPU, cheapest first
  python box.py launch      # rent the cheapest GPU offer; wait until ssh works; ids -> box_state.json
  python box.py status
  python box.py ssh -- <cmd>
  python box.py teardown    # delete the instance

Several boxes at once: BOX=<name> selects box_state_<name>.json (default box_state.json). GPU=<gpu_type> (default H100).
EXCLUDE_CLOUDS=<a,b> skips those clouds.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, "/srv/projects/unlearning-explorations/src/experimental_v6/2026.10.08/verda")
import shadeform_run as S  # noqa: E402

HERE = Path(__file__).resolve().parent
STATE = HERE / (f"box_state_{os.environ['BOX']}.json" if os.environ.get("BOX") else "box_state.json")
GPU = os.environ.get("GPU", "H100")   # Shadeform gpu_type, e.g. A100_80G


def offers():
    out = []
    for it in S.api("GET", "/instances/types").get("instance_types", []):
        cfg = it.get("configuration", {})
        if cfg.get("num_gpus") != 1 or cfg.get("gpu_type") != GPU:
            continue
        if it.get("cloud") in os.environ.get("EXCLUDE_CLOUDS", "").split(","):
            continue
        for a in it.get("availability") or []:
            if a.get("available"):
                out.append({"cloud": it.get("cloud"), "region": a.get("region"),
                            "shade_instance_type": it.get("shade_instance_type"),
                            "price": (it.get("hourly_price") or 9e9) / 100})
    return sorted(out, key=lambda o: o["price"])


def launch():
    assert not STATE.exists(), f"{STATE} exists: a box is already recorded"
    key_id, key_path = S.managed_key()
    name = f"graft-{os.environ.get('BOX', 'repro')}-{time.strftime('%m%d-%H%M')}"
    for o in offers():   # cheapest first; an offer listed as available can still be refused (e.g. 409)
        try:
            iid = S.api("POST", "/instances/create", {
                "cloud": o["cloud"], "region": o["region"], "shade_instance_type": o["shade_instance_type"],
                "shade_cloud": True, "name": name, "ssh_key_id": key_id})["id"]
            break
        except urllib.error.HTTPError as e:
            print("create refused:", e.code, e.read()[:300], json.dumps(o), flush=True)
    else:
        raise SystemExit("no offer accepted")
    st = {"offer": o, "instance_id": iid, "name": name, "key_path": str(key_path)}
    STATE.write_text(json.dumps(st, indent=1))
    print("instance created:", iid, json.dumps(o), flush=True)
    for _ in range(90):
        info = S.api("GET", f"/instances/{iid}/info")
        if info["status"] in ("active", "error"):
            break
        time.sleep(20)
    print("status", info["status"], info.get("ip"), flush=True)
    if info["status"] != "active":
        raise SystemExit(f"instance not active: {info.get('status_details')}")
    user = S.wait_ssh(key_path, info["ip"], ["shadeform", "ubuntu", "root"])
    assert user, "ssh never came up"
    st.update(ip=info["ip"], user=user)
    STATE.write_text(json.dumps(st, indent=1))
    print(f"ssh up: {user}@{info['ip']}", flush=True)


def ssh_cmd():
    st = json.loads(STATE.read_text())
    return S.ssh_base(st["key_path"], st["ip"], st["user"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "launch", "status", "ssh", "teardown"])
    ap.add_argument("rest", nargs="*")
    a = ap.parse_args()
    if a.cmd == "search":
        for o in offers()[:10]:
            print(o)
    elif a.cmd == "launch":
        launch()
    elif a.cmd == "status":
        st = json.loads(STATE.read_text())
        info = S.api("GET", f"/instances/{st['instance_id']}/info")
        print(json.dumps({k: info.get(k) for k in ("status", "ip", "ssh_user", "hourly_price")}))
    elif a.cmd == "ssh":
        sys.exit(subprocess.run(ssh_cmd() + [" ".join(a.rest)]).returncode)
    elif a.cmd == "teardown":
        st = json.loads(STATE.read_text())
        S.api("POST", f"/instances/{st['instance_id']}/delete")
        print("instance delete requested:", st["instance_id"])
        STATE.rename(STATE.with_suffix(".torn_down.json"))


if __name__ == "__main__":
    main()
