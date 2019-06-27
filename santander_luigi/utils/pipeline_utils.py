import pathlib


def posify(path: pathlib.Path) -> str:
    return path.as_posix()

