"""GCP infrastructure and ops configuration for the post-training container."""

# triggers a feature-review alert after this many consecutive failures to promote
MAX_CONSECUTIVE_REJECTIONS: int = 3
