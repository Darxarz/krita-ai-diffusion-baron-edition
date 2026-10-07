import os
import re
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from zipfile import ZipFile

feed_url = "https://orchestrion.su/baron-updates/stable.json"
max_package_bytes = 32 * 1024 * 1024


def version_key(version: str):
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)-baron\.(\d+)", version)
    if not match:
        raise ValueError("Invalid Baron version")
    return tuple(int(n) for n in match.groups())


def package_info(data: dict, current_version: str, feed: str):
    if data.get("schema") != 1 or data.get("edition") != "baron" or data.get("channel") != "stable":
        raise ValueError("Invalid Baron update channel")
    package = data.get("windows", {})
    version = package.get("version", "")
    version_key(version)
    url = urlsplit(package.get("url", ""))
    origin = urlsplit(feed)
    if (
        url.scheme != "https"
        or url.hostname != origin.hostname
        or url.port != origin.port
        or url.username
        or url.password
        or url.query
        or url.fragment
        or not url.path.startswith("/baron-updates/releases/")
        or not url.path.endswith(".zip")
        or ".." in url.path.split("/")
    ):
        raise ValueError("Invalid Baron package address")
    size = package.get("bytes")
    if type(size) is not int or not 0 < size <= max_package_bytes:
        raise ValueError("Invalid Baron package size")
    if not re.fullmatch(r"[a-f0-9]{64}", package.get("sha256", "")):
        raise ValueError("Invalid Baron package checksum")
    return package if version_key(version) > version_key(current_version) else None


def install_package(archive: Path, destination: Path, version: str):
    destination = destination.absolute()
    stage = archive.parent / "extracted"
    with ZipFile(archive) as package:
        members = package.infolist()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.filename)
            parts = path.parts
            if (
                not parts
                or path.is_absolute()
                or "\\" in member.orig_filename
                or any(
                    p in (".", "..") or ":" in p or p.startswith(".") or p.endswith((".", " "))
                    for p in parts
                )
                or parts[0] not in ("ai_diffusion", "ai_diffusion.desktop")
                or (parts[0] == "ai_diffusion.desktop" and len(parts) != 1)
                or stat.S_ISLNK(member.external_attr >> 16)
                or any(
                    p.split(".")[0].upper()
                    in {
                        "CON",
                        "PRN",
                        "AUX",
                        "NUL",
                        *(f"COM{i}" for i in range(1, 10)),
                        *(f"LPT{i}" for i in range(1, 10)),
                    }
                    for p in parts
                )
            ):
                raise ValueError("Unsafe plugin archive path")
            normalized = member.filename.rstrip("/").casefold()
            if normalized in names:
                raise ValueError("Duplicate plugin archive path")
            names.add(normalized)
            total += member.file_size
        if len(members) > 10000 or total > 128 * 1024 * 1024:
            raise ValueError("Plugin archive exceeds the size limit")
        source = package.read("ai_diffusion/__init__.py").decode("utf-8")
        if not re.search(r"__version__\s*=\s*[\"\']" + re.escape(version) + r"[\"\']", source):
            raise ValueError("Plugin archive version does not match the update")
        if "ai_diffusion.desktop" not in package.namelist():
            raise ValueError("Plugin archive is incomplete")
        for member in members:
            path = destination / member.filename
            if any(parent.is_symlink() for parent in (path, *path.parents)):
                raise ValueError("Cannot update through a symbolic link")
        package.extractall(stage)

    backup = destination / ".baron-update-backups" / uuid.uuid4().hex
    replaced: list[tuple[Path, Path | None]] = []
    try:
        for source_file in sorted(stage.rglob("*")):
            if not source_file.is_file():
                continue
            relative = source_file.relative_to(stage)
            target = destination / relative
            previous = None
            if target.exists():
                previous = backup / relative
                previous.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, previous)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Place the replacement on the destination volume before atomic rename.
            pending = target.with_name(target.name + ".baron-pending-" + uuid.uuid4().hex)
            try:
                shutil.copy2(source_file, pending)
                os.replace(pending, target)
            finally:
                pending.unlink(missing_ok=True)
            replaced.append((target, previous))
    except Exception:
        for target, previous in reversed(replaced):
            if previous is None:
                target.unlink(missing_ok=True)
            else:
                shutil.copy2(previous, target)
        raise
    return backup
