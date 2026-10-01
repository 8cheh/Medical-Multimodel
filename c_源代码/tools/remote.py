"""Minimal SSH/SFTP helper for the AutoDL box.

Credentials live OUTSIDE the repo in %TEMP%/vitaldb_creds.json:
    {"host": "...", "port": 22, "user": "root", "password": "..."}

Usage:
    python tools/remote.py exec "nvidia-smi"
    python tools/remote.py exec -f script.sh          # run a local script remotely
    python tools/remote.py put local remote
    python tools/remote.py get remote local

Env:
    VITALDB_SSH_TIMEOUT   seconds for connect/banner/auth (default 30). Raise it when
                          the box is loaded with training jobs and sshd is slow to
                          finish password auth.

The server host key is pinned on first use and verified on every later connection, so
a swapped key is refused instead of silently accepted.
"""
import json
import os
import sys
import tempfile
import time
from contextlib import suppress

from paramiko import AutoAddPolicy, HostKeys, RejectPolicy, SSHClient, SSHException

_TMPDIR = os.environ.get("TEMP") or tempfile.gettempdir()
CREDS = os.path.join(_TMPDIR, "vitaldb_creds.json")
KNOWN_HOSTS = os.path.join(_TMPDIR, "vitaldb_known_hosts")


def _timeout():
    raw = os.environ.get("VITALDB_SSH_TIMEOUT")
    if not raw:
        return 30
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


SSH_TIMEOUT = _timeout()


def load_creds(path=CREDS):
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read credentials at {path}: {type(exc).__name__}: {exc}") from exc
    try:
        return data["host"], int(data["port"]), data["user"], data["password"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"credentials at {path} need host/port/user/password: {exc}") from exc


def pin_host_key(cli, host, port, path=KNOWN_HOSTS):
    """Trust-on-first-use: record the server key so later runs can verify it."""
    transport = cli.get_transport()
    if transport is None:
        return
    key = transport.get_remote_server_key()
    keys = HostKeys()
    with suppress(OSError, ValueError):
        keys.load(path)
    with suppress(OSError):
        keys.add(f"[{host}]:{port}", key.get_name(), key)
        keys.save(path)
        os.chmod(path, 0o600)


def _host_key_mismatch(exc, pinned):
    if not pinned:
        return ""
    if "not found in known_hosts" not in str(exc):
        return ""
    return (f"\nThe pinned host key no longer matches (instance rebuilt?). "
            f"If so, delete {KNOWN_HOSTS} and retry.")


def connect():
    host, port, user, password = load_creds()
    cli = SSHClient()
    pinned = False
    # Only load when the file exists: load_host_keys() sets _host_keys_filename even
    # when the file is missing, which then makes AutoAddPolicy try to save_host_keys()
    # into a non-existent file and raise FileNotFoundError.
    if os.path.exists(KNOWN_HOSTS):
        with suppress(OSError, ValueError):
            cli.load_host_keys(KNOWN_HOSTS)
            pinned = len(cli.get_host_keys()) > 0

    cli.set_missing_host_key_policy(RejectPolicy() if pinned else AutoAddPolicy())
    failure = None
    try:
        cli.connect(
            host,
            port=port,
            username=user,
            password=password,
            timeout=SSH_TIMEOUT,
            banner_timeout=SSH_TIMEOUT,
            auth_timeout=SSH_TIMEOUT,
        )
    except SSHException as exc:
        failure = exc
    if failure is not None:
        raise SystemExit(
            f"SSH connect to {host}:{port} failed: {failure}"
            f"{_host_key_mismatch(failure, pinned)}"
        ) from failure

    if not pinned:
        pin_host_key(cli, host, port)
    return cli


def run(cli, cmd, timeout=3600):
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    op = args[0]
    cli = connect()
    try:
        if op == "exec":
            if args[1] == "-f":
                try:
                    with open(args[2], encoding="utf-8") as fh:
                        cmd = fh.read()
                except OSError as exc:
                    print(f"cannot read {args[2]}: {exc}", file=sys.stderr)
                    return 1
            else:
                cmd = args[1]
            t0 = time.time()
            rc, out, err = run(cli, cmd)
            print(out, end="")
            if err.strip():
                print("--- STDERR ---", file=sys.stderr)
                print(err, end="", file=sys.stderr)
            print(f"\n[rc={rc} dt={time.time() - t0:.1f}s]")
            return rc
        if op in ("put", "get"):
            sftp = cli.open_sftp()
            try:
                if op == "put":
                    sftp.put(args[1], args[2])
                else:
                    sftp.get(args[1], args[2])
            finally:
                sftp.close()
            print("ok")
            return 0
        print(f"unknown op {op}")
        return 1
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
