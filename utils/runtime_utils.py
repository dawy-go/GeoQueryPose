import os
import subprocess

def set_cuda_visible_devices(gpu_ids):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def get_git_identity(repo_dir):
    def run_git(args):
        try:
            return subprocess.check_output(
                ["git", "-C", repo_dir, *args],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
        except (FileNotFoundError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return ""

    branch = run_git(["branch", "--show-current"])
    if not branch:
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if branch == "HEAD":
        branch = "detached"

    short_commit = run_git(["rev-parse", "--short", "HEAD"])
    if not branch and not short_commit:
        return ""
    if not short_commit:
        return f"branch={branch}"
    if not branch:
        return f"commit={short_commit}"
    return f"branch={branch}, commit={short_commit}"


def append_git_identity_to_note(note, repo_dir):
    git_identity = get_git_identity(repo_dir)
    if not git_identity:
        return note or ""

    suffix = f"[git: {git_identity}]"
    note = note or ""
    if suffix in note:
        return note
    return f"{note} {suffix}".strip()
