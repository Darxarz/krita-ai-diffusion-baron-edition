from urllib.parse import urlsplit, urlunsplit

website_url = "https://orchestrion.ink"


def migrate_url(value: str):
    parts = urlsplit(value)
    if parts.scheme == "https" and parts.netloc.lower() in ("orchestrion.su", "orchestrion.su:443"):
        return urlunsplit(parts._replace(netloc="orchestrion.ink"))
    return value
