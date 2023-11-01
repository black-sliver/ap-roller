#!/usr/bin/python3

import os
import shutil
import subprocess
import sys
import tempfile
import time
import typing
from base64 import urlsafe_b64encode
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from statistics import mean
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

if typing.TYPE_CHECKING:
    import argparse
    import pathlib
    PathLike = Union[str, pathlib.Path]
else:
    PathLike = Any


__version__ = "0.1.4"
try:
    def get_rev():
        # noinspection PyUnresolvedReferences
        from git import Repo
        repo = Repo(search_parent_directories=True)
        sha = repo.head.commit.hexsha
        return repo.git.rev_parse(sha, short=True)
    __version__ += f"-{get_rev()}"
except ImportError:
    pass


is_windows = os.name == "nt"
allowed_repos = []


class APInstall(NamedTuple):
    path: PathLike
    venv: Optional[str] = None

    def __str__(self) -> str:
        return os.path.normpath(os.path.join(self.path, self.venv) if self.venv else self.path)

    @lru_cache
    def get_short_name(self, all: Optional[Tuple["APInstall"]] = None, min_segments: int = 1) -> str:
        """return a short path-like name that is unique across all"""
        if all is None:
            return str(self)
        others = [other for other in all if other != self]
        parts = str(self).split(os.sep)
        if self.venv:
            min_segments += 1
        min_segments = min(min_segments, len(parts))
        if not others:
            return os.path.join(*parts[-min_segments:])
        others_normalized = [str(other) for other in others]
        for n in range(len(parts) - min_segments, -1, -1):
            candidate = os.path.join(*parts[n:])
            for other in others_normalized:
                if other.endswith(candidate):
                    break
            else:
                return candidate
        return str(self)  # duplicate


def _b64hash(*args: Any) -> str:
    return urlsafe_b64encode(str(hash(*args)).encode("utf-8")).rstrip(b"=").decode("ascii")


def _enter_venv(install: PathLike, venv: Optional[str] = None, msg: Optional[str] = None) -> str:
    script = f'cd "{install}" &&'
    if is_windows and venv:
        script += f'{venv}\\Scripts\\activate.bat && '
        script += f'echo "{msg}" && echo "venv: %VIRTUAL_ENV%" && echo && '
    elif venv:
        script += f'source {venv}/bin/activate && '
        script += f'echo "{msg}"; echo "venv: $VIRTUAL_ENV" && echo && '
    else:
        script += f'echo "{msg}" && echo "w/o venv" && echo && '
    return script


def generate(ap: APInstall, seed: str, yamls: Iterable[PathLike], output_dir: Optional[PathLike],
             py_args: Optional[List[str]] = None, ap_args: Optional[Dict[str, str]] = None,
             timeout: int = 60) -> Optional[float]:
    install, venv = ap
    with tempfile.TemporaryDirectory(prefix="ap-in_") as tmpin:
        with tempfile.TemporaryDirectory(prefix="ap-out_") as tmpout:
            for yaml in yamls:
                shutil.copy(yaml, tmpin)
            args = f'--player_files_path "{tmpin}" --outputpath "{tmpout}" --seed "{seed}"'
            if ap_args:
                for key, val in ap_args.items():
                    args += f' --{key} "{val}"' if " " in val else f' --{key} {val}'
            script = _enter_venv(install, venv, "Starting generate")
            script += f'echo "" | python {" ".join(py_args) if py_args else ""} Generate.py {args}'
            start = time.monotonic()
            try:
                res = subprocess.run(script, shell=True, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as ex:
                res = subprocess.CompletedProcess(script, -1, "",
                                                  str(ex).replace("timed out after", "\ntimed out after"))
            end = time.monotonic()
            if res.returncode == 0 and "Done. Enjoy." in res.stdout:
                return end - start
            if output_dir:
                run_id = f"{_b64hash(ap)[:8]}_{seed}_{_b64hash(tuple(yamls))[:16]}"
                try:
                    with open(os.path.join(output_dir, f"fail_{run_id}.txt"), "w", encoding="utf-8") as f:
                        f.write(f"Failed run {run_id}\n\n"
                                f"install: {install}\n"
                                f"venv: {venv}\n"
                                f"seed: {seed}\n"
                                f"yamls: {yamls}\n\n")
                        f.write("-- stdout --\n\n")
                        f.write(res.stdout)
                        f.write("\n")
                        f.write("-- stderr --\n\n")
                        f.write(res.stderr)
                        f.write("\n")
                except Exception as e:
                    print(f"Could not write fail_{run_id}.txt: {e}")

            return None  # did not complete


def module_update(ap: APInstall, py_args: Optional[List[str]] = None) -> None:
    install, venv = ap
    script = _enter_venv(install, venv, "Running ModuleUpdate")
    script += f'echo "" | python {" ".join(py_args) if py_args else ""} ModuleUpdate.py -y'
    if subprocess.run(script, shell=True, timeout=60).returncode:
        raise Exception(f"Error running ModuleUpdate for '{ap}'")

def pyx_build(ap: APInstall, py_args: Optional[List[str]] = None) -> None:
    install, venv = ap
    script = _enter_venv(install, venv, "Running pyx builds")
    script += f'python {" ".join(py_args) if py_args else ""} -c "import NetUtils"'
    if subprocess.run(script, shell=True, timeout=60).returncode:
        raise Exception(f"Error running pyx builds for '{ap}'")

def update_settings(ap: APInstall, py_args: Optional[List[str]] = None) -> None:
    install, venv = ap
    script = _enter_venv(install, venv, "Update host.yaml")
    script += f'echo "" | python {" ".join(py_args) if py_args else ""} Launcher.py --update_settings'
    if subprocess.run(script, shell=True, timeout=60).returncode:
        raise Exception(f"Error updating host.yaml for '{ap}'")

def collect_yamls(mode, max_slots, include=None, exclude=None, limit=1000) -> List[Tuple[str, ...]]:
    """
    Create a list of yaml combinations to roll.

    :param mode:
      one of 'default', 'minimal', 'all', where
      default picks no _minimal.yaml files, minimal only picks _minimal.yaml files
    :param max_slots:
      maximum number of slots in each set
    :param include:
      list of games to include, None if all games should be included, using directory names
    :param exclude:
      list of games to exclude, None if no games should be excluded, using directory names
    :param limit:
      maximum number of total yaml combinations
    :return:
      list of yaml file tuples
    """
    from pathlib import Path
    from random import Random
    yaml_dir = Path(__file__).parent / "Players"
    game_dirs = sorted(
        [f.relative_to(os.getcwd()) for f in yaml_dir.iterdir()
         if f.is_dir()
         and (include is None or f.name in include)
         and (exclude is None or f.name not in exclude)])

    def match_minimal(f: str) -> bool:
        return f.endswith("_minimal.yaml")

    def match_default(f: str) -> bool:
        return not match_minimal(f)

    def match_all(_: str) -> bool:
        return True

    match = match_minimal if mode == "minimal" else match_all if mode == "all" else match_default
    game_yamls: Dict[str, List[str]] = dict(filter(lambda kv: kv[1], (
        (game.name, sorted([str(f) for f in game.iterdir() if f.is_file() and match(f.name)]))
        for game in game_dirs
        )))
    games = list(game_yamls.keys())
    game_count = len(games)
    res: List[Tuple[str, ...]] = []
    for game, yamls in game_yamls.items():
        count = len(yamls)
        # 1. solo - 1 yaml per game
        res.append((yamls[0],))
        # 2. inter-game - 2 yamls per game
        if max_slots > 1 and count > 1:
            res.append(tuple(sorted((yamls[1 % count], yamls[2 % count]))))

    # 3. cross-game - 2 games, 1 yaml each
    if max_slots > 1:
        for n, game1 in enumerate(game_yamls):
            yamls1 = game_yamls[game1]
            count1 = len(yamls1)
            game2 = games[(n + 1) % game_count]
            yamls2 = game_yamls[game2]
            count2 = len(yamls2)
            res.append(tuple(sorted((yamls1[3 % count1], yamls2[4 % count2]))))

    # collect all unused solo yamls for 5
    solo_candidates = []
    for game, yamls in game_yamls.items():
        for yaml in yamls:
            if (yaml,) not in res:
                solo_candidates.append((yaml,))

    rng = Random(0)
    rng.shuffle(solo_candidates)
    mixed_collisions = 0
    while len(res) < limit and (len(solo_candidates) > 0 or mixed_collisions < 20):
        # 4. pick max_slots random yamls
        mixed_set = set()
        for n in range(max_slots):
            game = games[rng.randrange(0, game_count)]
            yaml = game_yamls[game][rng.randrange(0, len(game_yamls[game]))]
            mixed_set.add(yaml)
        mixed = tuple(sorted(mixed_set))
        if len(mixed) > 1 and mixed not in res:
            res.append(mixed)
            mixed_collisions = 0
        else:
            mixed_collisions += 1
        # 5. pick random solo yaml
        if len(solo_candidates) > 0:
            res.append(solo_candidates.pop())

    return res


def _waste_cycles(x=1):
    """call this to waste some cpu cycles"""
    for n in range(0, 1000):
        x *= 2
    return x


def main(args: "argparse.Namespace"):
    from glob import glob
    import itertools
    from urllib.parse import urlparse
    print(f"ap-roller {__version__}")
    args.extra = list(filter(lambda f: not f.endswith("basepatch.sfc"),
                             itertools.chain(*[glob(pattern) for pattern in args.extra])))
    if args.verbose:
        print(f"args: {args}")
    aps = [APInstall(*arg.rsplit(':', 1)) for arg in args.aps
           if not arg.startswith('#') and not arg.startswith('https://')]
    repos = [ap for ap in args.aps if ap.startswith('https://')]
    for pr in map(lambda ap: int(ap[1:]), (ap for ap in args.aps if ap.startswith('#'))):
        import requests
        r = requests.get(f"https://api.github.com/repos/ArchipelagoMW/Archipelago/pulls/{pr}").json()
        repos.append(f"{r['head']['repo']['clone_url']}:{r['head']['ref']}")
    with tempfile.TemporaryDirectory(prefix="ap-repos_") as repos_dir:
        for repo in repos:
            if ':' in repo[6:]:
                repo, label = repo.rsplit(':', 1)
            else:
                label = None
            if repo not in allowed_repos:
                if input(f"Allow running code from '{repo}'? [y/N] ").lower() not in ("y", "yes"):
                    return 1
                allowed_repos.append(repo)
            url = urlparse(repo)
            repo_folder_name = f"{url.hostname.replace('.', '-')}_{url.path[1:-4].replace('/', '-')}_" \
                               f"{label.replace('/', '-')}"
            if not all(0x20 <= ord(c) < 127 and c not in ("/", "\\", "\"", "'", ":") for c in repo_folder_name):
                raise Exception(f"Bad character in repo '{repo_folder_name}'")
            repo_dir = os.path.join(repos_dir, repo_folder_name)
            os.makedirs(repo_dir, 0o700)
            python = sys.executable
            venv = "venv"
            subprocess.run(f'cd "{repo_dir}" && '
                           f'git init -q && git remote add origin "{repo}" && '
                           f'git fetch -q origin "{label}" --depth=2 && git reset --hard FETCH_HEAD &&'
                           f'{python} -m venv --upgrade-deps {venv}', shell=True)
            for pattern in args.extra:
                for src in glob(pattern):
                    dst = os.path.join(repo_dir, os.path.basename(src))
                    print(f"{src} -> {dst}")
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                    else:
                        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*.pyc', '.git*', '__pycache__'))
            aps.append(APInstall(repo_dir, venv))
        assert aps, "No valid AP installation selected"  # in practice, we should error out before reaching this
        roll(tuple(aps), args)
    return 0


def roll(aps, args: "argparse.Namespace"):
    import concurrent.futures
    import threading
    # NOTE: currently the yaml stage has to be identical between the APs, otherwise it's hard to compare w/ weights
    # TODO: weights-to-yaml pre-parse
    repeat = args.repeat
    if args.seeds:
        if '-' in args.seeds:
            range_start, range_end = map(int, args.seeds.split('-'))
            seeds = map(str, range(range_start, range_end + 1))
        else:
            seeds = args.seeds.split(',')
    else:
        seeds = map(str, range(1, 6))  # default 1..5
    timeout = args.timeout
    ap_args = {}
    if args.spoiler:
        ap_args["spoiler"] = args.spoiler
    py_args = ['-O'] if args.optimize else []
    for ap in aps:
        assert os.path.isdir(str(ap)), f"AP directory or venv does not exist: '{ap}'"
    output_dir = os.path.join("output", f"{datetime.now().strftime('%m_%d-%H_%M_%S')}_{os.getpid()}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputting to {output_dir}\n")
    results = {"stats": {
        "elapsed": 0.0,
        "finished_execs": 0,
        "finished_runs": 0,
        "success": {ap.get_short_name(aps): 0 for ap in aps},
        "failed": {ap.get_short_name(aps): 0 for ap in aps},
        "time": {},
    }, "runs": [], "failed": []}
    for ap in aps:
        assert os.path.isdir(str(ap)), "AP directory or venv does not exist: '{ap}'"
        print(ap.get_short_name(aps), end=": ")
        sys.stdout.flush()
        module_update(ap)
        pyx_build(ap)
        update_settings(ap)
    all_times: Dict[APInstall, List[float]] = {}
    yaml_combinations = collect_yamls(args.yamls, args.slots,
                                      ",".join(args.include).split(',') if args.include else None,
                                      ",".join(args.exclude).split(',') if args.exclude else None,
                                      args.limit)

    write_lock = threading.Lock()

    def block(t=.05):
        # a thread/future running for t to avoid some runs finishing faster than others
        # NOTE: if you have timed turbo, you may have to block longer at the beginning to get accurate numbers
        t1 = time.monotonic()
        t2 = t1
        while t2 - t1 < t:
            _waste_cycles()
            t2 = time.monotonic()

    def one(seed, yamls, verbose=False):
        times: Dict[APInstall, Optional[List[float]]] = {}
        gen_id = f"{seed}_{_b64hash(tuple(yamls))[:16]}"
        if verbose:
            print(f"{gen_id}: {seed}, {yamls}")
        for _ in range(0, repeat):
            for ap in aps:
                if times.get(ap, True) is not None:  # don't repeat failed runs
                    res = generate(ap, seed, yamls, output_dir, ap_args=ap_args, py_args=py_args, timeout=timeout)
                    write_lock.acquire()
                    try:
                        results["stats"]["finished_execs"] += 1
                        if res is None:
                            times[ap] = None
                            fail_id = f"{_b64hash(ap)[:8]}_{gen_id}"
                            results["failed"].append({
                                "id": fail_id,
                                "gen_id": gen_id,
                                "ap": ap.get_short_name(),
                                "yamls": yamls,
                                "seed": seed,
                                "log": f"fail_{fail_id}.txt",
                            })
                        else:
                            times.setdefault(ap, []).append(res)
                            all_times.setdefault(ap, []).append(res)
                    finally:
                        write_lock.release()
            # TODO: live update output
        write_lock.acquire()
        try:
            now = time.monotonic()
            results["runs"].append({
                "gen_id": gen_id,
                "yamls": yamls,
                "seed": seed,
                "time": {
                    ap.get_short_name(aps): {
                        "min": min(times[ap]),
                        "avg": mean(times[ap]),
                        "max": max(times[ap])
                    } if times[ap] else None
                    for ap in aps
                }
            })
            for ap in aps:
                if times[ap]:
                    results["stats"]["success"][ap.get_short_name(aps)] += 1
                else:
                    results["stats"]["failed"][ap.get_short_name(aps)] += 1
            results["stats"]["finished_runs"] += 1
            results["stats"]["elapsed"] = now - start
            results["stats"]["time"] = {
                ap.get_short_name(aps): {
                    "min": min(all_times[ap]),
                    "avg": mean(all_times[ap]),
                    "max": max(all_times[ap])
                } for ap in aps if ap in all_times
            }
            import json
            with open(os.path.join(output_dir, 'results.json'), 'w', encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            if not verbose:
                print(".", end="")
                sys.stdout.flush()
        finally:
            write_lock.release()

    def one_functor(seed, yaml, verbose=False):
        return lambda: one(seed, yaml, verbose)

    print('Generating...')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        start = time.monotonic()
        futures = []
        for n in range(args.threads):
            futures.append(executor.submit(block))  # don't let one gen start before another
        for s in seeds:
            for y in yaml_combinations:
                futures.append(executor.submit(one_functor(s, y, args.verbose)))
        for n in range(args.threads - 1):
            futures.append(executor.submit(lambda: block(1)))  # don't turbo the last remaining generations
        for future in futures:
            future.result()  # wait for all futures to finish so we get exceptions
    print('All done.')


if __name__ == "__main__":
    import argparse

    default_threads = max(1, (os.cpu_count() or 1) - 2)

    parser = argparse.ArgumentParser(description="Roll and compare AP seeds between one or more versions")
    parser.add_argument('aps', metavar='dir:venv|"#<PR>"|repo:branch', type=str, nargs='+',
                        help='AP source directory followed by :venv')
    parser.add_argument("--repeat", default=4, type=int, help="Repeat each roll N times for averaging")
    parser.add_argument("--threads", default=default_threads, type=int, help="NUmber of CPU threads to use")
    parser.add_argument("--yamls", default="default",
                        help="Change which yamls to consider. 'default', 'minimal' or 'all'.")
    parser.add_argument("--include", action="append",
                        help="Comma separated list of games to include, using directory names. Default: all")
    parser.add_argument("--exclude", action="append",
                        help="Comma separated list of games to exclude, using directory names. Default: none")
    parser.add_argument("--extra", action="append", default=["extra/*"],
                        help="Glob string of extra files and folders to copy to downloaded repos. Default: ./extra/*")
    parser.add_argument("--slots", default=3, type=int, help="Max slots to roll at the same time")
    parser.add_argument("--limit", default=1000, type=int, help="Total games to roll")
    parser.add_argument("--seeds", help="Comma separated list or 'start-stop' of seed numbers to roll.")
    parser.add_argument("--spoiler", help="Passed to generate. 0=None, 2=Full")
    parser.add_argument("--timeout", default=60, type=int, help="Time limit of each generation in seconds.")
    parser.add_argument("-O", "--optimize", action="store_true", help="Run python with -O")
    parser.add_argument('-v', '--verbose', action="store_true", help="Be more verbose")

    exit(main(parser.parse_args()))
