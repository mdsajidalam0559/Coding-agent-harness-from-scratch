"""Run the agent's shell commands inside a locked-down Docker container.

One long-lived container per session: the project is mounted at /workspace, nothing else from the
host is visible, secret files are masked, and the network is off unless explicitly enabled.
"""
import os
import subprocess
import time
import uuid

IMAGE = "agent-sandbox:latest"
DOCKERFILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sandbox")
MASKED_FILES = [".env"]


def _docker(*args, timeout=120, check=True):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, errors="replace",
                            stdin=subprocess.DEVNULL, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {args[0]} failed: {result.stderr.strip()}")
    return result


def require_docker():
    """Stop with instructions (not a traceback) when Docker cannot be used."""
    try:
        result = _docker("info", "--format", "{{.ServerVersion}}", check=False, timeout=20)
    except FileNotFoundError:
        raise SystemExit("Docker is not installed. Install it, or run with the sandbox off (see --help).")
    if result.returncode != 0:
        if "permission denied" in result.stderr.lower():
            raise SystemExit("Docker is installed, but your user may not use it. Run once:\n"
                             "    sudo usermod -aG docker $USER\n"
                             "then log out and back in (or reboot), and check with `docker ps`.\n"
                             "Until then you can use --sandbox off (commands run unsandboxed on this machine).")
        raise SystemExit(f"Docker is not usable: {result.stderr.strip()[:300]}")


def ensure_image():
    if _docker("image", "inspect", IMAGE, check=False).returncode != 0:
        print(f"Building sandbox image {IMAGE} (first run only)...")
        _docker("build", "-t", IMAGE, DOCKERFILE_DIR, timeout=900)


class DockerSandbox:
    def __init__(self, workspace, network=False, memory="1g", cpus="1", pids=256, mounts=()):
        self.workspace = os.path.realpath(workspace)
        self.mounts = list(mounts)  # (host_path, container_path) pairs, mounted read-only
        self.network = network
        self.limits = {"memory": memory, "cpus": cpus, "pids": pids}
        self.name = None

    def start(self):
        require_docker()
        ensure_image()
        self.name = f"agent-sandbox-{uuid.uuid4().hex[:8]}"
        args = [
            "run", "-d", "--rm", "--name", self.name,
            "--network", "bridge" if self.network else "none",
            "--memory", self.limits["memory"], "--memory-swap", self.limits["memory"],
            "--cpus", str(self.limits["cpus"]), "--pids-limit", str(self.limits["pids"]),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",  # files the agent creates stay owned by you
            "-e", "HOME=/tmp",
            "-v", f"{self.workspace}:/workspace", "-w", "/workspace",
        ]
        for host_path, container_path in self.mounts:
            args += ["-v", f"{os.path.realpath(host_path)}:{container_path}:ro"]
        for name in MASKED_FILES:
            if os.path.isfile(os.path.join(self.workspace, name)):
                args += ["-v", f"/dev/null:/workspace/{name}:ro"]  # the file looks empty inside
        _docker(*args, IMAGE, "sleep", "infinity")
        return self

    def exec(self, command, timeout, env):
        """Run command in the container. Returns (returncode, stdout, stderr, timed_out)."""
        env_args = [arg for key, value in env.items() for arg in ("-e", f"{key}={value}")]
        # GNU timeout kills the command's whole process group inside the container;
        # killing the `docker exec` client from the host would leave it running.
        argv = ["docker", "exec", *env_args, self.name,
                "timeout", "--kill-after=2", str(timeout), "bash", "-c", command]
        start = time.monotonic()
        try:
            result = subprocess.run(argv, capture_output=True, text=True, errors="replace",
                                    stdin=subprocess.DEVNULL, timeout=timeout + 15)
        except KeyboardInterrupt:
            self.interrupt()
            raise
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return None, out, err, True
        timed_out = result.returncode in (124, 137) and time.monotonic() - start >= timeout - 0.5
        return result.returncode, result.stdout, result.stderr, timed_out

    def interrupt(self):
        """Kill every process in the container except PID 1 (killing the docker exec client does not)."""
        _docker("exec", self.name, "kill", "-KILL", "-1", check=False, timeout=15)

    def stop(self):
        if self.name:
            _docker("rm", "-f", self.name, check=False, timeout=30)
            self.name = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
