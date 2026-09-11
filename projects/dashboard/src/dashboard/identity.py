"""Read the caller's identity from the Identity-Aware Proxy headers.

IAP terminates authentication at the edge, in front of Cloud Run, and forwards the verified
identity on X-Goog-Authenticated-User-Email. The container is not publicly reachable — only
IAP's service agent holds roles/run.invoker on it — so a request that arrives here has
already been authenticated and authorised by IAP.

This is therefore display-only: it answers "who am I signed in as" in the UI. It must never
become an access-control decision. The header is trivially forgeable by anything that can
reach the container directly, and the moment this app gates data on it, the security story
moves from "IAP, at the edge, enforced by IAM" to "a string comparison in Python". Real
per-user authorisation would verify the signed X-Goog-IAP-JWT-Assertion instead; the
dashboard shows one shared snapshot to everyone IAP lets through, so it needs neither.
"""

from __future__ import annotations

# IAP prefixes the value with the identity namespace, e.g. "accounts.google.com:a@b.com".
_IAP_EMAIL_HEADER = "X-Goog-Authenticated-User-Email"
_NAMESPACE_SEPARATOR = ":"


def viewer_email(headers: dict[str, str] | None) -> str | None:
    """Return the signed-in viewer's email address, or None when not behind IAP.

    Returning None is the expected case for local development, where the app is run without
    IAP in front of it; the UI degrades to not naming a viewer rather than refusing to render.
    """
    if not headers:
        return None

    # Header lookup is case-insensitive per RFC 9110, but Streamlit hands over a plain dict.
    lowered = {key.lower(): value for key, value in headers.items()}
    raw = lowered.get(_IAP_EMAIL_HEADER.lower())
    if not raw:
        return None

    _, separator, email = raw.partition(_NAMESPACE_SEPARATOR)
    return (email if separator else raw).strip() or None
