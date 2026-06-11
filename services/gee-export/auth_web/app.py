import json
import os
import webbrowser
from pathlib import Path

# Allow OAuth over HTTP for localhost development
os.environ.setdefault("OAUTHLIB_INSECURE_TRANSPORT", "1")

import google.auth.transport.requests
from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

PROJECT_ROOT = Path(__file__).parents[3]
TOKEN_PATH = PROJECT_ROOT / "config" / "gee_user_credentials.json"

load_dotenv(PROJECT_ROOT / ".env")

SCOPES = [
    "openid",
    "email",
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/devstorage.full_control",
]

CLIENT_ID = os.getenv("GEE_OAUTH_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("GEE_OAUTH_CLIENT_SECRET", "")

CLIENT_CONFIG = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost:5000/callback"],
    }
}

app = Flask(__name__)
app.secret_key = os.urandom(24)


def _save_credentials(creds):
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    with open(TOKEN_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return data


@app.route("/")
def index():
    has_token = TOKEN_PATH.exists()
    email = session.get("user_email")
    return render_template("index.html", has_token=has_token, email=email)


@app.route("/login")
def login():
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri="http://localhost:5000/callback",
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        prompt="consent",
        include_granted_scopes="true",
    )
    session["state"] = state
    return redirect(auth_url)


@app.route("/callback")
def callback():
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        state=session.get("state"),
        redirect_uri="http://localhost:5000/callback",
    )
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    data = _save_credentials(creds)

    email = "Akun Google Anda"
    if creds.id_token:
        try:
            payload = creds.id_token
            import jwt
            decoded = jwt.decode(payload, options={"verify_signature": False})
            email = decoded.get("email", email)
            session["user_email"] = email
        except Exception:
            pass

    print(f"[AUTH] Credentials saved → {TOKEN_PATH}")
    if email:
        print(f"[AUTH] Authenticated as: {email}")

    return render_template("success.html", email=email)


@app.route("/logout")
def logout():
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
    session.pop("user_email", None)
    return redirect(url_for("index"))


@app.route("/status")
def status():
    has_token = TOKEN_PATH.exists()
    email = session.get("user_email")
    return {"authenticated": has_token, "email": email}


def main():
    print("=" * 60)
    print("  Deforest.id — GEE Web Authentication")
    print("=" * 60)
    print()

    if not CLIENT_ID or not CLIENT_SECRET:
        print("  [!] GEE_OAUTH_CLIENT_ID dan GEE_OAUTH_CLIENT_SECRET")
        print("      belum diatur di file .env!")
        print()
        print("  Cara setup:")
        print("  1. Buka https://console.cloud.google.com/apis/credentials")
        print("  2. Buat OAuth 2.0 Client ID (tipe: Web Application)")
        print("  3. Tambah redirect URI: http://localhost:5000/callback")
        print("  4. Copy Client ID dan Client Secret ke .env:")
        print()
        print("     GEE_OAUTH_CLIENT_ID=xxx.apps.googleusercontent.com")
        print("     GEE_OAUTH_CLIENT_SECRET=GOCSPX-xxx")
        print()
        print("  Setelah itu, jalankan ulang perintah ini.")
        print()
        return

    if TOKEN_PATH.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
            email = session.get("user_email", "terautentikasi")
            print(f"  [\u2713] Sudah login sebagai: {email}")
            print(f"  Token: {TOKEN_PATH}")
            print()
            print("  Ingin ganti akun? Buka http://localhost:5000/logout")
            print()
        except Exception:
            pass

    print(f"  Buka http://localhost:5000 di browser untuk login.")
    print()
    webbrowser.open("http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
