mod firebase_client;
mod rag;

use std::{
    collections::HashMap,
    env,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use firebase_auth::{FirebaseAuth, FirebaseUser};
use firebase_client::{AuthError, FirebaseClient, Session};
use rag::{Hit, Rag, RagError};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ------------------------------------------------------------------- state

/// One message of the chat history shown in the browser.
#[derive(Serialize, Clone)]
struct Turn {
    role: &'static str, // "user" | "assistant"
    content: String,
    sources: Vec<Hit>,
}

struct AppState {
    rag: Rag,
    fb: FirebaseClient,
    verifier: FirebaseAuth,
    /// Chat history per signed-in user, in memory only (no database).
    /// Plays the role of `st.session_state`; cleared at sign-out.
    histories: Mutex<HashMap<String, Vec<Turn>>>,
}

type Shared = Arc<AppState>;

// ------------------------------------------------------------------- errors

struct ApiError(StatusCode, String);

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": self.1 }))).into_response()
    }
}

impl From<AuthError> for ApiError {
    fn from(e: AuthError) -> Self {
        match e {
            AuthError::InvalidInput(m) => ApiError(StatusCode::BAD_REQUEST, m),
            AuthError::Rejected(m) => ApiError(StatusCode::UNAUTHORIZED, m),
            AuthError::Network(m) => ApiError(StatusCode::BAD_GATEWAY, m),
        }
    }
}

impl From<RagError> for ApiError {
    fn from(e: RagError) -> Self {
        let code = match e {
            RagError::EmptyQuestion => StatusCode::BAD_REQUEST,
            RagError::OllamaUnreachable(_) => StatusCode::SERVICE_UNAVAILABLE,
            RagError::Model(_) => StatusCode::BAD_GATEWAY,
        };
        ApiError(code, e.to_string())
    }
}

/// Only signed-in users pass: verifies the Firebase ID token from `Authorization: Bearer ...`.
fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<FirebaseUser, ApiError> {
    let token = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .ok_or_else(|| ApiError(StatusCode::UNAUTHORIZED, "Please sign in first.".into()))?;

    state.verifier.verify::<FirebaseUser>(token).map_err(|_| {
        ApiError(
            StatusCode::UNAUTHORIZED,
            "Your session is invalid or expired. Please sign in again.".into(),
        )
    })
}

// ------------------------------------------------------------------ handlers

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[derive(Deserialize)]
struct Credentials {
    email: String,
    password: String,
}

async fn sign_up(State(s): State<Shared>, Json(c): Json<Credentials>) -> Result<Json<Session>, ApiError> {
    Ok(Json(s.fb.sign_up(&c.email, &c.password).await?))
}

async fn sign_in(State(s): State<Shared>, Json(c): Json<Credentials>) -> Result<Json<Session>, ApiError> {
    Ok(Json(s.fb.sign_in(&c.email, &c.password).await?))
}

/// Clears the chat history of the user. Succeeds even if the token already expired.
async fn sign_out(State(s): State<Shared>, headers: HeaderMap) -> Json<serde_json::Value> {
    if let Ok(user) = authenticate(&s, &headers) {
        s.histories.lock().unwrap().remove(&user.user_id);
    }
    Json(json!({ "ok": true }))
}

#[derive(Deserialize)]
struct ChatRequest {
    question: String,
}

#[derive(Serialize)]
struct ChatResponse {
    answer: String,
    sources: Vec<Hit>,
    generated: bool,
}

async fn chat(
    State(s): State<Shared>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let user = authenticate(&s, &headers)?;
    let question = req.question.trim().to_string();

    let ans = s.rag.answer(&question).await?;

    {
        let mut all = s.histories.lock().unwrap();
        let h = all.entry(user.user_id.clone()).or_default();
        h.push(Turn { role: "user", content: question, sources: vec![] });
        h.push(Turn {
            role: "assistant",
            content: ans.answer.clone(),
            sources: ans.hits.clone(),
        });
    }

    Ok(Json(ChatResponse {
        answer: ans.answer,
        sources: ans.hits,
        generated: ans.generated,
    }))
}

// ---------------------------------------------------------------------- main

fn required(name: &str) -> String {
    env::var(name).unwrap_or_else(|_| {
        eprintln!("Missing {name} in .env (see .env.example)");
        std::process::exit(1);
    })
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();

    let api_key = required("FIREBASE_API_KEY");
    let project_id = required("FIREBASE_PROJECT_ID");
    let cfg = rag::RagConfig::from_env().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    println!("Embedding the travel guide with {} ...", cfg.embed_model);
    let rag = Rag::build(cfg, "knowledge_base.json").await.unwrap_or_else(|e| {
        eprintln!("Start-up failed: {e}");
        std::process::exit(1);
    });
    println!(
        "Guide ready: {} entries (top-k = {}, min score = {})",
        rag.entry_count(),
        rag.config().top_k,
        rag.config().min_score
    );

    let state = Arc::new(AppState {
        rag,
        fb: FirebaseClient::new(api_key),
        verifier: FirebaseAuth::new(&project_id).await,
        histories: Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/api/signup", post(sign_up))
        .route("/api/signin", post(sign_in))
        .route("/api/signout", post(sign_out))
        .route("/api/chat", post(chat))
        .with_state(state);

    let port = env::var("PORT").unwrap_or_else(|_| "8501".into());
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap_or_else(|e| {
        eprintln!("Cannot listen on {addr}: {e}");
        std::process::exit(1);
    });
    println!("Open http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}