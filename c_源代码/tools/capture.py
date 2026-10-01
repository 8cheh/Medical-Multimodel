"""Run a local shell script on the remote box and save its stdout as UTF-8.

PowerShell's Tee-Object writes UTF-16LE, which silently corrupts evidence files
that later steps read back as UTF-8. This helper talks to the box through
paramiko and always writes plain UTF-8.

Usage:
    python tools/capture.py results/out.txt tools/some_script.sh
    python tools/capture.py results/out.txt -- "date; nvidia-smi"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import remote  # noqa: E402


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 1
    out_path, target = argv[1], argv[2]

    if target == "--":
        if len(argv) < 4:
            print("missing command after --")
            return 1
        cmd = argv[3]
    else:
        try:
            with open(target, encoding="utf-8") as fh:
                cmd = fh.read()
        except OSError as exc:
            print(f"cannot read {target}: {exc}", file=sys.stderr)
            return 1

    cli = remote.connect()
    try:
        rc, out, err = remote.run(cli, cmd)
    finally:
        cli.close()

    try:
        with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(out)
            if err.strip():
                fh.write("\n--- STDERR ---\n")
                fh.write(err)
    except OSError as exc:
        print(f"cannot write {out_path}: {exc}", file=sys.stderr)
        return 1

    encoding = "utf-8"
    try:
        with open(out_path, "rb") as fh:
            head = fh.read(4)
        if head.startswith(b"\xff\xfe"):
            encoding = "utf-16le (UNEXPECTED)"
    except OSError:
        pass
    print(f"wrote {out_path} rc={rc} bytes={os.path.getsize(out_path)} encoding={encoding}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv))
