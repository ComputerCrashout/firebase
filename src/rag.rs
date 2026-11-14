//! Retrieval-augmented generation over `knowledge_base.json`.
//!
//! Pipeline for one question:
//!   embed question -> score every entry (cosine) -> filter / top-k
//!   -> build context -> ask the chat model -> return answer + sources.

use std::{cmp::Ordering, collections::HashSet, env, time::Duration};

use ollama_rs::{
    Ollama,
    generation::{
        chat::{ChatMessage, request::ChatMessageRequest},
        embeddings::request::{EmbeddingsInput, GenerateEmbeddingsRequest},
    },
};
use serde::{Deserialize, Serialize};
use url::Url;

pub const IDK: &str = "I don't know. The travel guide does not contain an answer to that.";

const SYSTEM_PROMPT: &str = "You are the travel assistant of Lumen Travel for Ho Chi Minh City. \
Answer ONLY with facts found in the CONTEXT entries. Never use outside knowledge and never guess \
opening hours, prices or distances. After each fact, cite the entry ID in square brackets, e.g. [G03]. \
If the CONTEXT does not contain the answer, reply exactly: I don't know. \
Be brief (at most 4 sentences) and answer in the language of the question.";

// ---------------------------------------------------------------- data types

/// One entry of the travel guide (abstraction: only the fields we need).
#[derive(Deserialize, Clone, Debug)]
pub struct Entry {
    pub id: String,
    pub title: String,
    pub text: String,
    pub source: String,
    /// Only places have a category / tags (used again by RAG-Rec).
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

struct Indexed {
    entry: Entry,
    embedding: Vec<f32>,
}
 
/// A retrieved entry with its cosine score.
#[derive(Serialize, Clone, Debug)]
pub struct Hit {
    pub id: String,
    pub title: String,
    pub text: String,
    pub source: String,
    pub category: Option<String>,
    pub score: f32,
    /// `true` when the score reached MIN_SCORE, so the entry is given to the model.
    pub used: bool,
}

#[derive(Serialize, Clone, Debug)]
pub struct Answer {
    pub answer: String,
    pub hits: Vec<Hit>,
    /// `false` when we answered "I don't know" without calling the chat model.
    pub generated: bool,
}

#[derive(Debug)]
pub enum RagError {
    EmptyQuestion,
    OllamaUnreachable(String),
    Model(String),
}

#[derive(Clone, Debug)]
pub struct RagConfig {
    pub ollama_url: String,
    pub embed_model: String,
    pub chat_model: String,
    pub top_k: usize,
    pub min_score: f32,
}

impl RagConfig {
    pub fn from_env() -> Result<Self, String> {
        // Local Ollama by default; no tunnel needed.
        let ollama_url = env::var("OLLAMA_URL")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .unwrap_or_else(|| "http://localhost:11434".to_string());
        Ok(Self {
            ollama_url,
            embed_model: env::var("EMBED_MODEL").unwrap_or_else(|_| "embeddinggemma".into()),
            chat_model: env::var("CHAT_MODEL").unwrap_or_else(|_| "qwen3:4b".into()),
            top_k: env::var("TOP_K").ok().and_then(|v| v.parse().ok()).unwrap_or(3),
            min_score: env::var("MIN_SCORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.40),
        })
    }
}

// ----------------------------------------------------------------------- Rag

pub struct Rag {
    ollama: Ollama,
    http: reqwest::Client,
    cfg: RagConfig,
    index: Vec<Indexed>,
}

impl Rag {
    /// Loads the guide and embeds every entry ONCE (at start-up).
    pub async fn build(cfg: RagConfig, kb_path: &str) -> Result<Self, String> {
        let raw = std::fs::read_to_string(kb_path)
            .map_err(|e| format!("Cannot read {kb_path}: {e}"))?;
        let entries: Vec<Entry> =
            serde_json::from_str(&raw).map_err(|e| format!("{kb_path} is not valid: {e}"))?;
        validate(&entries)?;

        let url = Url::parse(&cfg.ollama_url).map_err(|e| format!("OLLAMA_URL is not a valid URL: {e}"))?;
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(8))
            .build()
            .map_err(|e| e.to_string())?;

        let mut rag = Rag {
            ollama: Ollama::from(url),
            http,
            cfg,
            index: Vec::new(),
        };

        rag.check_server().await.map_err(|e| format!("{e}"))?;
        rag.ensure_models().await?;

        let docs: Vec<String> = entries.iter().map(doc_text).collect();
        let res = rag
            .ollama
            .generate_embeddings(GenerateEmbeddingsRequest::new(
                rag.cfg.embed_model.clone(),
                EmbeddingsInput::Multiple(docs),
            ))
            .await
            .map_err(|e| {
                format!(
                    "Embedding the guide failed ({e}). Is `{}` pulled on the Ollama server?",
                    rag.cfg.embed_model
                )
            })?;

        if res.embeddings.len() != entries.len() {
            return Err("Ollama returned a different number of embeddings than guide entries".into());
        }
        rag.index = entries
            .into_iter()
            .zip(res.embeddings)
            .map(|(entry, embedding)| Indexed { entry, embedding })
            .collect();
        Ok(rag)
    }

    /// Pulls the embedding and chat models if the local Ollama does not have them yet.
    async fn ensure_models(&self) -> Result<(), String> {
        let local: Vec<String> = self
            .ollama
            .list_local_models()
            .await
            .map_err(|e| format!("Could not list Ollama models: {e}"))?
            .into_iter()
            .map(|m| m.name)
            .collect();

        for wanted in [&self.cfg.embed_model, &self.cfg.chat_model] {
            let have = local.iter().any(|n| {
                n == wanted || n == &format!("{wanted}:latest") || wanted.strip_suffix(":latest") == Some(n.as_str())
            });
            if !have {
                println!("Model `{wanted}` not found locally, pulling it (first run only, this can take a while)...");
                self.ollama
                    .pull_model(wanted.clone(), false)
                    .await
                    .map_err(|e| format!("Could not pull `{wanted}`: {e}"))?;
                println!("Model `{wanted}` ready.");
            }
        }
        Ok(())
    }

    pub fn entry_count(&self) -> usize {
        self.index.len()
    }

    pub fn config(&self) -> &RagConfig {
        &self.cfg
    }

    /// Answers one question. Steps: embed -> retrieve -> build context -> generate.
    pub async fn answer(&self, question: &str) -> Result<Answer, RagError> {
        let question = question.trim();
        if question.is_empty() {
            return Err(RagError::EmptyQuestion);
        }

        // 1. embed
        let q_emb = self.embed(&format!("task: search result | query: {question}")).await?;

        // 2. retrieve (filter -> score -> top-k)
        let hits = self.retrieve(&q_emb, |_| true);

        // 3. "I don't know" rule: nothing close enough -> do not even call the model.
        if !hits.iter().any(|h| h.used) {
            return Ok(Answer { answer: IDK.into(), hits, generated: false });
        }

        // 4. build context from the entries that passed the threshold
        let context = hits
            .iter()
            .filter(|h| h.used)
            .map(|h| format!("[{}] {}\n{}", h.id, h.title, h.text))
            .collect::<Vec<_>>()
            .join("\n\n");
        let user_msg = format!("CONTEXT:\n{context}\n\nQUESTION: {question}\n/no_think");

        // 5. generate
        let req = ChatMessageRequest::new(
            self.cfg.chat_model.clone(),
            vec![ChatMessage::system(SYSTEM_PROMPT.to_string()), ChatMessage::user(user_msg)],
        );
        let resp = match self.ollama.send_chat_messages(req).await {
            Ok(r) => r,
            Err(e) => return Err(self.classify(e.to_string()).await),
        };
        let text = strip_think(&resp.message.content);
        let answer = if text.is_empty() { IDK.to_string() } else { text };

        Ok(Answer { answer, hits, generated: true })
    }

    /// Reusable "filter -> score -> top-k" step. RAG-Rec can pass a category filter here.
    pub fn retrieve(&self, query_emb: &[f32], filter: impl Fn(&Entry) -> bool) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .index
            .iter()
            .filter(|ix| filter(&ix.entry)) // filter
            .map(|ix| {
                let score = cosine(query_emb, &ix.embedding); // score
                Hit {
                    id: ix.entry.id.clone(),
                    title: ix.entry.title.clone(),
                    text: ix.entry.text.clone(),
                    source: ix.entry.source.clone(),
                    category: ix.entry.category.clone(),
                    score,
                    used: score >= self.cfg.min_score,
                }
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        hits.truncate(self.cfg.top_k); // top-k
        hits
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>, RagError> {
        let req = GenerateEmbeddingsRequest::new(
            self.cfg.embed_model.clone(),
            EmbeddingsInput::Single(text.to_string()),
        );
        match self.ollama.generate_embeddings(req).await {
            Ok(r) => r
                .embeddings
                .into_iter()
                .next()
                .ok_or_else(|| RagError::Model("The embedding model returned nothing".into())),
            Err(e) => Err(self.classify(e.to_string()).await),
        }
    }

    /// Ping the server: unreachable server vs. model problem.
    async fn check_server(&self) -> Result<(), RagError> {
        let url = format!("{}/api/tags", self.cfg.ollama_url.trim_end_matches('/'));
        match self.http.get(url).send().await {
            Ok(r) if r.status().is_success() => Ok(()),
            Ok(r) => Err(RagError::OllamaUnreachable(format!(
                "The Ollama server answered with HTTP {}",
                r.status()
            ))),
            Err(e) => Err(RagError::OllamaUnreachable(format!("{e}"))),
        }
    }

    async fn classify(&self, err: String) -> RagError {
        match self.check_server().await {
            Err(unreachable) => unreachable,
            Ok(()) => RagError::Model(err),
        }
    }
}

impl std::fmt::Display for RagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RagError::EmptyQuestion => write!(f, "Please type a question."),
            RagError::OllamaUnreachable(e) => write!(
                f,
                "The Ollama server is not reachable. Start it with `ollama serve` (or open the Ollama app) and check OLLAMA_URL. ({e})"
            ),
            RagError::Model(e) => write!(f, "The model returned an error: {e}"),
        }
    }
}

// ------------------------------------------------------------------- helpers

fn validate(entries: &[Entry]) -> Result<(), String> {
    if !(12..=20).contains(&entries.len()) {
        return Err(format!(
            "knowledge_base.json must have 12-20 entries (found {})",
            entries.len()
        ));
    }
    let mut seen = HashSet::new();
    for e in entries {
        if !seen.insert(e.id.as_str()) {
            return Err(format!("Duplicate entry id {}", e.id));
        }
        if e.title.trim().is_empty() || e.text.trim().is_empty() {
            return Err(format!("Entry {} has an empty title or text", e.id));
        }
    }
    Ok(())
}

/// embeddinggemma document format: "title: ... | text: ...".
fn doc_text(e: &Entry) -> String {
    let mut t = format!("title: {} | text: {}", e.title, e.text);
    if let Some(c) = &e.category {
        t.push_str(&format!(" | category: {c}"));
    }
    if !e.tags.is_empty() {
        t.push_str(&format!(" | tags: {}", e.tags.join(", ")));
    }
    t
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

/// qwen3 may emit `<think>...</think>`; remove it before showing the answer.
fn strip_think(s: &str) -> String {
    let mut out = s.to_string();
    while let (Some(start), Some(end)) = (out.find("<think>"), out.find("</think>")) {
        if end < start {
            break;
        }
        out.replace_range(start..end + "</think>".len(), "");
    }
    if let Some(pos) = out.rfind("</think>") {
        out = out[pos + "</think>".len()..].to_string();
    }
    out.trim().to_string()
}