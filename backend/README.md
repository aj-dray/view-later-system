## Testing

```bash
.venv/bin/python -m pytest
```

## Access tokens

Use the helper script to mint a JWT for service integrations. It will create the user on first run if requested.

```bash
../later-system/.venv/bin/python3 tools/generate_access_token.py <username> --ensure-user
```

Provide `--expires-in-hours` to override the default expiry or `--password` to set a specific password for a newly created user. The script loads environment variables from `.env`, so ensure `BACKEND_SECRET` is configured before issuing production tokens.
