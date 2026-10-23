# Deployment notes

The Streamlit configuration in `.streamlit/config.toml` sets a 10 MB upload
limit, hides detailed application errors, enables XSRF/CORS protections, and
disables browser usage-stat collection. These are baseline settings, not a
security or compliance certification.

Before exposing the app:

- Put it behind HTTPS and an authenticated access gateway; neither is provided
  by this project.
- Supply any required credentials through the deployment platform's secret
  manager. Do not commit `.env` files or `.streamlit/secrets.toml`.
- For multiple app instances, configure a strong shared Streamlit cookie secret
  through the deployment environment/secret manager. Never put its value in
  this repository.
- Review host, proxy, backup, and monitoring retention policies for uploaded
  files and generated/transcribed content. The app's clear action removes
  session-held app data; it cannot clear data retained by infrastructure.
- Recheck the upload limits, resource usage, and model checkpoints for the
  target environment before launch.

No legal compliance, medical accuracy, or deployment security claim is made.
