"""Render a settings object as the container environment variables it is read back from.

Vertex AI Pipelines containers have no Terraform to set their environment the way the Cloud
Run jobs do — the pipeline spec *is* their deployment descriptor. Pinning the settings onto
each task's container at compile time closes that gap in both directions:

* a deployment can retune the platform through ``ML_*``/``MODELING_*`` variables on the
  Cloud Build worker, and the next compiled pipeline carries them, no code change; and
* the policy a given run used is recorded in the pipeline spec itself, so a promotion six
  months ago can be explained by the gate that was actually in force at the time rather
  than by whatever the defaults say today.
"""

import json

from pydantic_settings import BaseSettings


def settings_env(settings: BaseSettings) -> dict[str, str]:
    """Return the env-var name/value pairs that reproduce ``settings`` in another process.

    Names are rebuilt from the settings class's own ``env_prefix``, so they round-trip
    through the same reader that produced them. Structured fields are JSON-encoded, which
    is exactly how pydantic-settings parses a complex field back out of the environment.
    """
    prefix = settings.model_config.get("env_prefix") or ""
    return {
        f"{prefix}{name}".upper(): value if isinstance(value, str) else json.dumps(value)
        for name, value in settings.model_dump(mode="json").items()
    }
