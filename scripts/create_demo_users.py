"""Create or update private Supabase demo users for local/team testing.

This script never exposes credentials in the frontend. It requires the
Supabase service-role key and should only be run by the project owner.
"""

import os
from pathlib import Path

import httpx


def load_root_env() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name}. Put it in the root .env temporarily before running this script.")
    return value


def create_or_update_user(
    client: httpx.Client,
    base_url: str,
    service_role_key: str,
    email: str,
    password: str,
    name: str,
    role: str,
) -> str:
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
    }
    response = client.post(
        f"{base_url}/auth/v1/admin/users",
        headers=headers,
        json={
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"name": name, "role": role},
        },
    )
    if response.status_code in {400, 422} and "already" in response.text.lower():
        users_response = client.get(
            f"{base_url}/auth/v1/admin/users",
            headers=headers,
            params={"page": 1, "per_page": 1000},
        )
        users_response.raise_for_status()
        existing = next(
            (
                item
                for item in users_response.json().get("users", [])
                if item.get("email", "").lower() == email.lower()
            ),
            None,
        )
        if not existing:
            raise RuntimeError(
                f"{email} already exists, but could not be found through the Supabase admin API."
            )
        user_id = existing["id"]
        update_response = client.put(
            f"{base_url}/auth/v1/admin/users/{user_id}",
            headers=headers,
            json={
                "password": password,
                "email_confirm": True,
                "user_metadata": {"name": name, "role": role},
            },
        )
        update_response.raise_for_status()
        print(f"Updated {role} demo user: {email} ({user_id})")
        return user_id

    response.raise_for_status()
    user_id = response.json()["id"]
    print(f"Created {role} demo user: {email} ({user_id})")
    return user_id


def set_role(
    client: httpx.Client,
    base_url: str,
    service_role_key: str,
    user_id: str,
    email: str,
    role: str,
) -> None:
    response = client.post(
        f"{base_url}/rest/v1/user_profiles",
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        },
        json={"id": user_id, "email": email, "role": role},
    )
    response.raise_for_status()


def main() -> None:
    load_root_env()
    base_url = required("SUPABASE_URL").rstrip("/")
    service_role_key = required("SUPABASE_SERVICE_ROLE_KEY")
    individual_email = required("DEMO_INDIVIDUAL_EMAIL")
    individual_password = required("DEMO_INDIVIDUAL_PASSWORD")
    recruiter_email = required("DEMO_RECRUITER_EMAIL")
    recruiter_password = required("DEMO_RECRUITER_PASSWORD")

    with httpx.Client(timeout=30) as client:
        individual_id = create_or_update_user(
            client,
            base_url,
            service_role_key,
            individual_email,
            individual_password,
            "Demo Individual",
            "individual",
        )
        recruiter_id = create_or_update_user(
            client,
            base_url,
            service_role_key,
            recruiter_email,
            recruiter_password,
            "Demo Recruiter",
            "recruiter",
        )
        set_role(client, base_url, service_role_key, individual_id, individual_email, "individual")
        set_role(client, base_url, service_role_key, recruiter_id, recruiter_email, "recruiter")

    print("Demo users are ready. Remove the service-role key and demo passwords after running.")


if __name__ == "__main__":
    main()
