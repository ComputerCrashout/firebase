//! Firebase Authentication through the public REST API (Identity Toolkit).
//!
//! Sign-up and sign-in go through this client. Verifying the returned ID token
//! on protected routes is done in `main.rs` with the `firebase-auth` crate.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const BASE: &str = "https://identitytoolkit.googleapis.com/v1";

#[derive(Clone)]
pub struct FirebaseClient {
    http: Client,
    api_key: String,
}

/// What the browser keeps after a successful sign-in / sign-up.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Session {
    pub id_token: String,
    pub email: String,
}

#[derive(Debug)]
pub enum AuthError {
    /// Empty email / password (caught before calling Firebase).
    InvalidInput(String),
    /// Firebase said no (wrong password, email already used, ...).
    Rejected(String),
    /// Could not talk to Firebase at all.
    Network(String),
}

impl FirebaseClient {
    pub fn new(api_key: String) -> Self {
        Self {
            http: Client::new(),
            api_key,
        }
    }

    pub async fn sign_up(&self, email: &str, password: &str) -> Result<Session, AuthError> {
        self.call("accounts:signUp", email, password).await
    }

    pub async fn sign_in(&self, email: &str, password: &str) -> Result<Session, AuthError> {
        self.call("accounts:signInWithPassword", email, password).await
    }

    async fn call(&self, endpoint: &str, email: &str, password: &str) -> Result<Session, AuthError> {
        let email = email.trim();
        if email.is_empty() || password.is_empty() {
            return Err(AuthError::InvalidInput(
                "Please enter both email and password.".into(),
            ));
        }

        let url = format!("{BASE}/{endpoint}?key={}", self.api_key);
        let resp = self
            .http
            .post(url)
            .json(&json!({
                "email": email,
                "password": password,
                "returnSecureToken": true
            }))
            .send()
            .await
            .map_err(|e| AuthError::Network(format!("Could not reach Firebase: {e}")))?;

        let ok = resp.status().is_success();
        let body: Value = resp
            .json()
            .await
            .map_err(|e| AuthError::Network(format!("Unexpected Firebase response: {e}")))?;

        if ok {
            serde_json::from_value(body)
                .map_err(|e| AuthError::Network(format!("Unexpected Firebase response: {e}")))
        } else {
            let code = body["error"]["message"].as_str().unwrap_or("UNKNOWN");
            Err(AuthError::Rejected(friendly(code)))
        }
    }
}

fn friendly(code: &str) -> String {
    match code {
        "EMAIL_EXISTS" => "This email is already registered. Try signing in.".into(),
        "EMAIL_NOT_FOUND" | "INVALID_PASSWORD" | "INVALID_LOGIN_CREDENTIALS" => {
            "Wrong email or password.".into()
        }
        "INVALID_EMAIL" => "That email address is not valid.".into(),
        "USER_DISABLED" => "This account has been disabled.".into(),
        c if c.starts_with("WEAK_PASSWORD") => "Password is too weak (use at least 6 characters).".into(),
        c if c.starts_with("TOO_MANY_ATTEMPTS") => "Too many attempts. Please try again later.".into(),
        "API_KEY_INVALID" | "INVALID_API_KEY" => "The Firebase API key in .env is not valid.".into(),
        other => format!("Firebase error: {other}"),
    }
}