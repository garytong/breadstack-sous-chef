#!/bin/bash
# The training VM's startup script (infra/envs/train/vm.tf): runs as root at
# every boot, through the guest agent; its output is in the VM's journal
# (sudo journalctl -u google-startup-scripts). It prepares the machine for
# the lab's training (docs/TRAIN-WORKSTATION.md, the same recipe) and leaves
# it idle for SSH:
#
#   1. the shared environment (/etc/sous-chef-train.env), the `souschef` user
#      that owns everything under /work, apt tools (make, git, rsync; cmake,
#      g++ and libcurl for export.py's llama.cpp build), uv pinned
#   2. the lab kit: `git clone` / `pull` of sous-chef-lab-repo when set, else
#      the bucket's kit/lab.tar.gz (make -C lab train-remote-push), unpacked
#      into /work/sous-chef/lab — venv, data and runs beside it are kept
#   3. lab/.venv-train, exactly as `make -C lab train-env` builds it (uv,
#      Python 3.12, train-constraints.txt, torch from the cu130 index)
#   4. the base model's PUBLIC weights into /work/hf (HF_HOME): the bucket's
#      base/<slug>-hf.tar when there is one (in-region, no NAT charge), else
#      Hugging Face through Cloud NAT; checked shard by shard, once per disk
#   5. `make -C lab train-gpu-check` (import unsloth on the GPU; never yet run
#      on a GPU before this VM) → /work/gpu-check.txt
#
# Steps 2-5 run in a transient systemd unit, sous-chef-bootstrap, so this
# script returns at once and SSH works while they run:
# `sous-chef-train-status` on the VM says where they are. The dataset is
# NEVER copied here: `sous-chef-data-pull <set>` does, by hand, after login.
set -uo pipefail

md() { curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
bucket=$(md sous-chef-bucket) || { echo "sous-chef: no sous-chef-bucket in the metadata"; exit 1; }
uv_version=$(md sous-chef-uv-version) || uv_version=0.12.19
base_model=$(md sous-chef-base-model) || base_model=
lab_repo=$(md sous-chef-lab-repo) || lab_repo=
lab_ref=$(md sous-chef-lab-ref) || lab_ref=main
[ -n "$lab_ref" ] || lab_ref=main

work=/work
lab=$work/sous-chef/lab
hf=$work/hf

# Where things are, for every login shell. No credentials: gcloud and the
# Google libraries use the VM's own account through the metadata server.
# HF_HOME: one Hugging Face cache for the VM, filled below.
# expandable_segments: less fragmentation of the GPU's memory between rows of
# very different lengths (the lab's rows run from ~2k to ~16k tokens).
cat > /etc/sous-chef-train.env <<EOF
SOUS_CHEF_TRAIN_BUCKET=gs://$bucket
SOUS_CHEF_TRAIN_WORK=$work
SOUS_CHEF_LAB=$lab
SOUS_CHEF_BASE_MODEL=$base_model
HF_HOME=$hf
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
chmod 0644 /etc/sous-chef-train.env

cat > /etc/profile.d/sous-chef-train.sh <<'EOF'
set -a; . /etc/sous-chef-train.env; set +a
export PATH="/usr/local/bin:$PATH"
case $- in *i*) [ "$(id -un)" = souschef ] || echo "sous-chef: the lab, its venv and the HF cache belong to souschef: sudo -iu souschef   (sous-chef-train-status for the bootstrap's state)";; esac
EOF
chmod 0644 /etc/profile.d/sous-chef-train.sh

# One owner for the lab, the venv, the HF cache and the runs: the system
# user souschef, home /work/souschef (uv's cache, ~/.unsloth's llama.cpp,
# ~/.triton). Operators arrive by OS Login as themselves and `sudo -iu
# souschef` (make -C lab train-vm-ssh does so).
install -d -m 0755 $work
id souschef >/dev/null 2>&1 || useradd --system --create-home --home-dir $work/souschef --shell /bin/bash souschef
install -d -m 0755 -o souschef -g souschef $work/sous-chef $hf $hf/hub

# Tools the lab needs that the image may lack: make (the lab's targets), git
# and rsync (the kit), cmake / g++ / python3.12-dev / libcurl (Unsloth builds
# llama.cpp for the GGUF export when it cannot fetch a prebuilt bundle from
# GitHub).
if ! command -v make >/dev/null || ! command -v rsync >/dev/null || ! command -v cmake >/dev/null; then
  # The image's package lists are frozen at image build, so install without an
  # update can 404 on superseded versions; the lock timeout rides out
  # unattended-upgrades holding dpkg on first boot.
  DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 update -q >/dev/null \
    && DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=300 install -y -q --no-install-recommends make git rsync cmake build-essential python3.12-dev libcurl4-openssl-dev >/dev/null \
    || echo "sous-chef: apt-get failed (Cloud NAT? the egress rule?): make / cmake may be missing"
fi

# The DLVM's gcloud is the snap build (/snap/bin), which systemd units and
# runuser shells do not have on PATH: without this, every bucket check in the
# helpers below "fails" silently (command not found behind 2>/dev/null) and
# the bootstrap reports "no kit" while the object is there (measured
# 2026-10-04 on the first boot).
for tool in gcloud gsutil; do
  [ -x /usr/local/bin/$tool ] || { [ -x /snap/bin/$tool ] && ln -sf /snap/bin/$tool /usr/local/bin/$tool; }
done
command -v gcloud >/dev/null || echo "sous-chef: no gcloud on PATH — the bucket helpers will fail"
# ...and snap confinement refuses to run for a user whose home is outside
# /home ("home directories outside of /home needs configuration"): souschef
# lives in /work, so register it (idempotent; measured on the same boot).
if [ -x /snap/bin/gcloud ] && command -v snap >/dev/null; then
  snap set system homedirs=/work 2>/dev/null || echo "sous-chef: 'snap set system homedirs=/work' failed — gcloud will refuse to run as souschef"
fi

# uv, pinned, for everyone (the lab's Python 3.12 and PyTorch come through
# it, pinned by lab/train-constraints.txt). Once per version.
if [ "$(/usr/local/bin/uv --version 2>/dev/null | awk '{print $2}')" != "$uv_version" ]; then
  curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh \
    || echo "sous-chef: installing uv $uv_version failed (Cloud NAT? the egress rule?)"
fi

# The GPU, as the driver sees it. A Secure Boot VM whose image carries an
# unsigned NVIDIA module would fail here.
if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
  echo "sous-chef: GPU ready"
else
  echo "sous-chef: nvidia-smi failed: the NVIDIA driver is not loaded (Secure Boot on? a new image family?)"
fi

# --- helpers for the operator (all run as souschef) ---------------------------

# sous-chef-kit-pull: the lab's code, fresh. From the git remote when one is
# configured, else the bucket's kit/lab.tar.gz (the working tree the
# workstation pushed, prefix lab/). The tar is unpacked beside the tree and
# rsync'd over it, so lab/.venv-train, lab/data and lab/runs survive a pull.
cat > /usr/local/bin/sous-chef-kit-pull <<EOF
#!/bin/bash
set -uo pipefail
lab=$lab; bucket=$bucket; repo='$lab_repo'; ref='$lab_ref'
if [ -n "\$repo" ]; then
  if [ -d "\$lab/.git" ] || [ -d "\$(dirname "\$lab")/.git" ]; then
    git -C "\$(dirname "\$lab")" fetch --quiet origin && git -C "\$(dirname "\$lab")" checkout --quiet "\$ref" && git -C "\$(dirname "\$lab")" pull --quiet --ff-only origin "\$ref" \
      && echo "sous-chef: lab at \$(git -C "\$(dirname "\$lab")" rev-parse --short HEAD) (\$ref) from \$repo" || { echo "sous-chef: git pull FAILED"; exit 1; }
  else
    git clone --quiet --branch "\$ref" "\$repo" "\$(dirname "\$lab")" && echo "sous-chef: cloned \$repo (\$ref) into \$(dirname "\$lab")" || { echo "sous-chef: git clone FAILED"; exit 1; }
  fi
else
  tmp=\$(mktemp -d) || exit 1
  trap 'rm -rf "\$tmp"' EXIT
  obj=\$(gcloud storage ls "gs://\$bucket/kit/lab.tar.gz" 2>/dev/null || gcloud storage ls "gs://\$bucket/kit/lab-*.tar" 2>/dev/null | tail -n 1)
  [ -n "\$obj" ] || { echo "sous-chef: no gs://\$bucket/kit/lab.tar.gz (make -C lab train-remote-push on the workstation; a git-archive kit/lab-<sha>.tar is taken too)"; exit 1; }
  case "\$obj" in *.gz|*.tgz) z=-z;; *) z=;; esac
  gcloud storage cat "\$obj" | tar -x \$z -C "\$tmp" || { echo "sous-chef: \$obj is not a tar"; exit 1; }
  [ -d "\$tmp/lab" ] || { echo "sous-chef: \$obj has no lab/ at its top"; exit 1; }
  mkdir -p "\$lab"
  rsync -a --delete --exclude=.venv-train --exclude=.venv --exclude=data --exclude=runs --exclude=unsloth_compiled_cache --exclude=__pycache__ --exclude='*.egg-info' "\$tmp/lab/" "\$lab/" \
    && echo "sous-chef: lab from \$obj → \$lab (\$(cat "\$lab/KIT-VERSION" 2>/dev/null || echo 'no KIT-VERSION'))" || { echo "sous-chef: rsync of the kit FAILED"; exit 1; }
fi
EOF

# sous-chef-data-pull <set>: the dataset the workstation pushed (data/<set>/
# in the bucket) → lab/data/<set>/. By hand, after login: the bootstrap never
# copies it.
cat > /usr/local/bin/sous-chef-data-pull <<EOF
#!/bin/bash
set -uo pipefail
[ \$# -eq 1 ] && echo "\$1" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*\$' || { echo "usage: sous-chef-data-pull <set>   (gcloud storage ls gs://$bucket/data/ lists them)"; exit 1; }
mkdir -p "$lab/data/\$1" && gcloud storage rsync --recursive "gs://$bucket/data/\$1" "$lab/data/\$1" \
  && echo "sous-chef: data/\$1 → $lab/data/\$1 (\$(du -sh "$lab/data/\$1" | cut -f1)); train with: make -C $lab train DATA=data/\$1 NAME=<run>"
EOF

# sous-chef-runs-push <name>: an adapter and/or export the VM made
# (lab/runs/adapters/<name>, lab/runs/exports/<name>) → the bucket's runs/,
# for make -C lab train-remote-pull NAME=<name> on the workstation.
cat > /usr/local/bin/sous-chef-runs-push <<EOF
#!/bin/bash
set -uo pipefail
[ \$# -eq 1 ] && echo "\$1" | grep -Eq '^[A-Za-z0-9][A-Za-z0-9._-]*\$' || { echo "usage: sous-chef-runs-push <name>   (a folder under $lab/runs/adapters/ and/or runs/exports/)"; exit 1; }
n=0
for kind in adapters exports; do
  d="$lab/runs/\$kind/\$1"; [ -d "\$d" ] || continue
  gcloud storage rsync --recursive --exclude='^gguf/.*' "\$d" "gs://$bucket/runs/\$kind/\$1" && echo "sous-chef: runs/\$kind/\$1 (\$(du -sh --exclude=gguf "\$d" | cut -f1)) → gs://$bucket/runs/\$kind/\$1" && n=\$((n + 1))
done
[ \$n -gt 0 ] || { echo "sous-chef: nothing named \$1 under $lab/runs/adapters or runs/exports"; exit 1; }
echo "sous-chef: on the workstation: make -C lab train-remote-pull NAME=\$1"
EOF

# sous-chef-train-status: the bootstrap's state, the GPU, the disk.
cat > /usr/local/bin/sous-chef-train-status <<EOF
#!/bin/bash
echo "bootstrap: \$(systemctl is-active sous-chef-bootstrap 2>/dev/null || echo inactive) — \$(cat $work/BOOTSTRAP 2>/dev/null || echo 'not finished yet (journalctl -u sous-chef-bootstrap -f)')"
echo "gpu:       \$(nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'nvidia-smi failed')"
echo "gpu-check: \$(cat $work/gpu-check.txt 2>/dev/null | tail -n 1 || echo 'not run yet')"
echo "lab:       $lab (\$(cat $lab/KIT-VERSION 2>/dev/null || echo 'no KIT-VERSION')); venv: \$([ -x $lab/.venv-train/bin/python ] && echo ok || echo missing)"
echo "base:      \$([ -f $hf/hub/models--${base_model//\//--}/.sous-chef-complete ] && echo "$base_model in $hf" || echo "$base_model not whole yet (journalctl -u sous-chef-bootstrap)")"
echo "data:      \$(ls -d $lab/data/*/ 2>/dev/null | xargs -rn1 basename | paste -sd' ' - || true) \$([ -d $lab/data ] && [ -n "\$(ls -A $lab/data 2>/dev/null)" ] || echo 'none (sous-chef-data-pull <set>)')"
echo "runs:      \$(ls -d $lab/runs/adapters/*/ 2>/dev/null | xargs -rn1 basename | paste -sd' ' - || echo none)"
echo "disk:      \$(df -h $work | awk 'NR == 2 { print \$4 " free of " \$2 }')"
echo "stops at:  \$(curl -fsS -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/scheduling/termination-timestamp 2>/dev/null || echo "max_run_duration (vm.tf), or make -C lab train-vm-down")"
EOF
chmod 0755 /usr/local/bin/sous-chef-kit-pull /usr/local/bin/sous-chef-data-pull /usr/local/bin/sous-chef-runs-push /usr/local/bin/sous-chef-train-status

# --- the base model's cache check -------------------------------------------
# A cache counts as there only with its .sous-chef-complete marker AND every
# shard whole. The marker is written after the fetch and the check, never
# taken from the bucket's tar (--exclude below), so a copy cut short by a
# preemption is fetched again at the next boot.
cat > /usr/local/sbin/sous-chef-check-base <<'CHECK'
#!/usr/bin/python3
# sous-chef-check-base <cache>: exit 0 when the cached base model is whole. In
# each snapshot, every shard the index names is there and exactly as long as
# its own header says (8 bytes, the header, the data), and the shards' data
# add up to the index's total_size (no index: its *.safetensors, the same
# way). Reads the headers only, in seconds. Prints counts, nothing else.
import glob, json, os, struct, sys


def data_bytes(path):
    """The shard's data length when the file is whole, else None."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if 8 + n > size:
                return None
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, struct.error):
        return None
    return end if size == 8 + n + end else None


snaps = sorted(glob.glob(os.path.join(sys.argv[1], "snapshots", "*", "")))
shards = whole = bad = missing = off_total = 0
ok = bool(snaps)
for snap in snaps:
    index = os.path.join(snap, "model.safetensors.index.json")
    total = None
    try:
        if os.path.lexists(index):
            with open(index) as f:
                meta = json.load(f)
            names = sorted(set(meta["weight_map"].values()))
            if "total_size" in meta.get("metadata", {}):
                total = int(meta["metadata"]["total_size"])
        else:
            names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(snap, "*.safetensors")))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        bad += 1
        ok = False
        continue
    ok = ok and bool(names)
    shards += len(names)
    data = 0
    for name in names:
        if not os.path.exists(os.path.join(snap, name)):
            missing += 1
            continue
        n = data_bytes(os.path.join(snap, name))
        if n is None:
            bad += 1
        else:
            whole += 1
            data += n
    if total is not None and data != total:
        off_total += 1
ok = ok and bad == 0 and missing == 0 and off_total == 0
print(f"sous-chef: base check: snapshots={len(snaps)} shards={shards} whole={whole} bad={bad} missing={missing}"
      f" off_total={off_total}: {'whole' if ok else 'NOT whole'}")
sys.exit(0 if ok else 1)
CHECK
chmod 0755 /usr/local/sbin/sous-chef-check-base

# sous-chef-fetch-base <bucket> <base_model> <hf_home>: the base weights into
# <hf_home>/hub, from the bucket's base/<slug>-hf.tar when there is one
# (make -C lab base-tar's layout, hub/models--…/ relative to HF_HOME; a tar
# with models--…/ at its top is taken too: both are unpacked into a staging
# folder beside the cache and the model folder moved into place), else from
# Hugging Face through Cloud NAT. Checked shard by shard; the marker is
# written only then. Runs as souschef.
cat > /usr/local/sbin/sous-chef-fetch-base <<'FETCH'
#!/bin/bash
set -uo pipefail
bucket=$1 base_model=$2 hf=$3
slug=$(echo "$base_model" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
model_dir="models--${base_model//\//--}"
cache="$hf/hub/$model_dir"
in="$hf/.incoming"
rm -rf "$cache" "$in"
if gcloud storage objects describe "gs://$bucket/base/$slug-hf.tar" --format='value(name)' >/dev/null 2>&1; then
  echo "sous-chef: base $base_model from gs://$bucket/base/$slug-hf.tar (in-region)"
  mkdir -p "$in" && gcloud storage cat "gs://$bucket/base/$slug-hf.tar" | tar -x -C "$in" --exclude=.sous-chef-complete || { rm -rf "$in"; echo "sous-chef: the base tar did not unpack"; exit 1; }
  src=
  for c in "$in/hub/$model_dir" "$in/$model_dir"; do [ -d "$c" ] && { src=$c; break; }; done
  [ -n "$src" ] || { rm -rf "$in"; echo "sous-chef: base/$slug-hf.tar holds no $model_dir (neither hub/$model_dir nor $model_dir at its top)"; exit 1; }
  mv "$src" "$cache" && rm -rf "$in"
else
  echo "sous-chef: base $base_model from Hugging Face (through Cloud NAT; make -C lab train-remote-push caches it in the bucket)"
  HF_HOME=$hf /usr/local/bin/uvx --from 'huggingface_hub[hf_xet]' hf download "$base_model" --quiet
fi && /usr/local/sbin/sous-chef-check-base "$cache" && touch "$cache/.sous-chef-complete" && echo "sous-chef: base model $base_model ready in $hf" \
  || { rm -rf "$cache"; echo "sous-chef: fetching $base_model FAILED, the partial copy cleared"; exit 1; }
FETCH
chmod 0755 /usr/local/sbin/sous-chef-fetch-base

# --- the bootstrap: kit, venv, base, GPU check, as souschef -------------------
# The base fetch (sous-chef-fetch-base) runs beside the venv build; both
# must end before the GPU check.
cache="$hf/hub/models--${base_model//\//--}"
cat > /usr/local/sbin/sous-chef-bootstrap <<EOF
#!/bin/bash
set -uo pipefail
set -a; . /etc/sous-chef-train.env; set +a
export PATH=/usr/local/bin:\$PATH
rm -f $work/BOOTSTRAP
as_souschef() { runuser -u souschef -- env HF_HOME=$hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/usr/local/bin:\$PATH "\$@"; }
fail=

# 4 (in the background): the base weights.
base_rc=0
if [ -n "$base_model" ]; then
  if [ -f "$cache/.sous-chef-complete" ] && /usr/local/sbin/sous-chef-check-base "$cache"; then
    echo "sous-chef: base model $base_model already in $hf"
  else
    [ ! -e "$cache" ] || echo "sous-chef: the base model's copy in $hf is partial (a fetch cut short): fetching it again"
    as_souschef /usr/local/sbin/sous-chef-fetch-base "$bucket" "$base_model" "$hf" &
    base_pid=\$!
  fi
fi

# 2: the kit.
as_souschef /usr/local/bin/sous-chef-kit-pull || fail="\$fail kit"

# 3: the venv, as make -C lab train-env builds it (a no-op when it is there).
if [ -z "\$fail" ]; then
  as_souschef make --no-print-directory -C $lab train-env && echo "sous-chef: $lab/.venv-train ready" || fail="\$fail venv"
fi

[ -z "\${base_pid:-}" ] || { wait "\$base_pid" || fail="\$fail base"; }

# 5: import unsloth on the GPU (the first time anywhere on a GPU for this venv).
if [ -z "\$fail" ]; then
  as_souschef make --no-print-directory -C $lab train-gpu-check 2>&1 | tee $work/gpu-check.txt | tail -n 1
  [ "\${PIPESTATUS[0]}" -eq 0 ] && echo "sous-chef: train-gpu-check ok" || fail="\$fail gpu-check"
fi

if [ -z "\$fail" ]; then
  echo "ready \$(date -u +%FT%TZ): kit, .venv-train, base and train-gpu-check ok — sudo -iu souschef; sous-chef-data-pull <set>" > $work/BOOTSTRAP
  echo "sous-chef: READY — no dataset was copied (sous-chef-data-pull <set> does); the bucket is gs://$bucket"
else
  echo "FAILED \$(date -u +%FT%TZ):\$fail — journalctl -u sous-chef-bootstrap; rerun (the unit is transient and gone once it exits): sudo systemd-run --unit=sous-chef-bootstrap --collect --property=User=root /usr/local/sbin/sous-chef-bootstrap" > $work/BOOTSTRAP
  echo "sous-chef: bootstrap FAILED:\$fail"
  exit 1
fi
EOF
chmod 0755 /usr/local/sbin/sous-chef-bootstrap

if systemctl is-active --quiet sous-chef-bootstrap; then
  echo "sous-chef: the bootstrap is running already (journalctl -u sous-chef-bootstrap -f)"
else
  systemd-run --unit=sous-chef-bootstrap --collect --property=User=root /usr/local/sbin/sous-chef-bootstrap >/dev/null 2>&1 \
    && echo "sous-chef: bootstrap started in the background (journalctl -u sous-chef-bootstrap -f; sous-chef-train-status)" \
    || echo "sous-chef: could not start the bootstrap (sudo systemctl reset-failed sous-chef-bootstrap; sudo /usr/local/sbin/sous-chef-bootstrap)"
fi

echo "sous-chef: startup done — SSH is open through IAP (make -C lab train-vm-ssh); the bucket is gs://$bucket"
