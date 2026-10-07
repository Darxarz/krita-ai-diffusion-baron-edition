import ctypes
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def install(bundle: Path, report_path: Path):
    import winreg

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("public_bundle") or manifest.get("personal_bundle"):
        raise ValueError("Only the reviewed public font bundle can be installed")
    rows = manifest["files"]
    for row in rows:
        for name, digest in [
            (row["file"], row["sha256"]),
            (row["license_file"], row["license_sha256"]),
        ]:
            path = bundle / name
            if not path.resolve().is_relative_to(bundle.resolve()):
                raise ValueError("Invalid bundle path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Bundle verification failed: {name}")
    target = Path(os.environ["LOCALAPPDATA"]) / "Microsoft" / "Windows" / "Fonts"
    target.mkdir(parents=True, exist_ok=True)
    receipts = target / "Baron-licenses"
    shutil.copytree(bundle / "licenses", receipts, dirs_exist_ok=True)
    shutil.copy2(bundle / "manifest.json", receipts / "manifest.json")
    gdi = ctypes.WinDLL("gdi32", use_last_error=True)
    gdi.AddFontResourceExW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint, ctypes.c_void_p]
    gdi.AddFontResourceExW.restype = ctypes.c_int
    report = []
    registry = r"Software\Microsoft\Windows NT\CurrentVersion\Fonts"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, registry) as key:
        for row in rows:
            path = target / ("Baron-" + row["file"])
            display = f"{row['font_family']} {row['font_subfamily']} (TrueType)"
            try:
                existing, _ = winreg.QueryValueEx(key, display)
            except FileNotFoundError:
                existing = None
            if existing and Path(existing) != path:
                report.append({
                    "family": row["family"],
                    "status": "preserved_existing",
                    "registry": existing,
                })
                continue
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                raise ValueError(f"Refusing to overwrite a different installed font: {path}")
            shutil.copy2(bundle / row["file"], path)
            winreg.SetValueEx(key, display, 0, winreg.REG_SZ, str(path))
            registered = gdi.AddFontResourceExW(str(path), 0, None)
            if not registered:
                raise OSError(f"Windows rejected font: {row['family']}")
            report.append({
                "family": row["family"],
                "status": "installed",
                "file": str(path),
                "sha256": row["sha256"],
                "registered_faces": registered,
                "registry_name": display,
            })
    user = ctypes.WinDLL("user32", use_last_error=True)
    user.SendMessageTimeoutW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_size_t,
        ctypes.c_ssize_t,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    result = ctypes.c_size_t()
    user.SendMessageTimeoutW(0xFFFF, 0x1D, 0, 0, 2, 1500, ctypes.byref(result))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({
            "installed": sum(r["status"] == "installed" for r in report),
            "preserved": sum(r["status"] == "preserved_existing" for r in report),
            "report": str(report_path),
        })
    )


if __name__ == "__main__":
    install(Path(sys.argv[1]), Path(sys.argv[2]))
