# EDA Agent Docker Sandbox

Docker is the only security backend for model-authored Python. macOS Seatbelt
and host-subprocess execution are intentionally unsupported.

## Build and verify

```bash
docker build -t eda-agent-sandbox:py312 docker/eda-agent-sandbox
uv run python scripts/check_sandbox.py
```

`requirements.in` is deliberately limited to libraries CodeAgent may import.
`requirements.lock` pins its transitive closure and every distribution hash;
regenerate it with the command recorded in the lock header. The Dockerfile also
pins its base image digest and installs with `--require-hashes`. At execution
time the broker resolves the configured image tag to an immutable local image
ID and passes that ID to `docker run`, so a tag change cannot alter an in-flight
execution.

Set `EDA_SANDBOX_REQUIRED=1` when application startup must fail unless the live
kernel-policy canary succeeds. Without that flag, deterministic EDA and
read-only SQL remain available, but every open Python request still fails
closed if the proof is unavailable.

## Enforced boundary

- Linux-container Docker engine with seccomp and cgroup namespaces
- model-authored code runs as non-root UID `65532`, with zero effective Linux
  capabilities and `no-new-privileges`
- PID 1 is a root supervisor with only `CAP_KILL`; after the analysis process
  exits it kills residual sandbox-UID processes before output adoption
- no network, private cgroup namespace, no IPC namespace sharing
- read-only root filesystem; `/tmp` is a bounded `noexec,nosuid,nodev` tmpfs
- pids, memory plus swap, CPU, wall-clock, file-descriptor, stdout, and stderr limits
- per-file, total-size, file-count, symlink, and special-file output checks
- model script mounted read-only at `/sandbox/analysis.py`
- one fresh byte- and inode-bounded `/work` tmpfs per execution
- only requested regular-file inputs, copied first to a private staging area and
  mounted read-only under `/work/inputs`
- after quiescence, trusted output adoption runs as the tmpfs owner and uses a
  container-side bounded tar stream followed by a second host-side
  path/type/count/size validation; the host output directory is never writable
  by live model-authored code
- scrubbed Docker CLI environment: application/API credentials do not enter the
  sandbox process
- SHA-256 input, code, stdout, stderr, image, policy, and output manifests stored
  outside the container-visible directory

The preflight canary verifies the effective analysis runtime—not just generated
command arguments—by checking UID, effective capabilities, `NoNewPrivs`,
seccomp, read-only root behavior, cgroup limits, and network denial.

Never mount the project root, workspace root, `.env`, credential directories, a
home directory, or the Docker socket into this image. The packaged `docker/app`
deployment intentionally provides no open-Python sandbox rather than exposing
the host Docker daemon; a production container deployment needs a separately
authenticated runner service or stronger runtime such as gVisor/microVM.
