from dashboard.identity import viewer_email


def test_strips_the_iap_identity_namespace():
    headers = {"X-Goog-Authenticated-User-Email": "accounts.google.com:someone@example.com"}
    assert viewer_email(headers) == "someone@example.com"


def test_header_lookup_is_case_insensitive():
    headers = {"x-goog-authenticated-user-email": "accounts.google.com:someone@example.com"}
    assert viewer_email(headers) == "someone@example.com"


def test_accepts_a_bare_address_without_a_namespace():
    assert viewer_email({"X-Goog-Authenticated-User-Email": "someone@example.com"}) == (
        "someone@example.com"
    )


def test_returns_none_when_not_behind_iap():
    # The local-development case: the app must still render, just without naming a viewer.
    assert viewer_email({}) is None
    assert viewer_email(None) is None
    assert viewer_email({"X-Goog-Authenticated-User-Email": ""}) is None
    assert viewer_email({"X-Goog-Authenticated-User-Email": "accounts.google.com:"}) is None
