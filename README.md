# ap-roller

Utility to test and compare Archipelago versions.

**Windows functionality is untested. Let me know or send a PR.**

Pass one or more `path/to/ap:venv` on command line after the flags.

By default this will generate 5000 games (5 Seeds * 1000 Player folders), 4 times per AP install,
taking ~2hrs per AP install, making ~5GB of disk io in ./output/ and a random amount in /tmp (~2GB).

Use e.g. `--limit 500`, `--seeds "1,2"`, `--repeat 1` to reduce that,
use `--exclude game1,game2` or `--include game1,game2` to get better cover with reduced limit.

Check `python ap-roller.py --help` and check [Players/README.md](Players/README.md) for more info.
