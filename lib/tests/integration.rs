//! Integration tests against real LLM providers.
//!
//! These tests are **skipped by default** — they only run when the
//! corresponding API key environment variables are set.
//!
//! Setup:
//! 1. Copy `.env.example` to `.env` in the repo root
//! 2. Fill in real API keys
//! 3. Run: `cargo test --manifest-path lib/Cargo.toml --test integration`
//!
//! Each test uses a simple prompt to verify end-to-end streaming works.

use agentive::*;
use std::sync::{Arc, Mutex};

/// Helper: skip test if env var is not set.
macro_rules! require_env {
    ($var:expr) => {
        match std::env::var($var) {
            Ok(val) if !val.is_empty() && !val.starts_with("sk-...") && val != "..." => val,
            _ => {
                eprintln!("Skipping: {} not set", $var);
                return;
            }
        }
    };
}

#[tokio::test]
async fn test_openai_simple_chat() {
    let api_key = require_env!("OPENAI_API_KEY");
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let provider = Arc::new(
        OpenAiProvider::new("https://api.openai.com/v1", &api_key, &model)
            .with_context_budget(50_000),
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let result = run(
        provider,
        vec![
            ChatMessage::system("You are a helpful assistant. Reply in one short sentence."),
            ChatMessage::user("What is 2 + 2?"),
        ],
        vec![],
        |_| Ok("unused".into()),
        RunnerConfig {
            max_iterations: 1,
            ..Default::default()
        },
        CancellationToken::new(),
        Steering::new(),
        Guardrails::default(),
        move |event| {
            events_clone.lock().unwrap().push(format!("{:?}", event));
        },
    )
    .await
    .expect("OpenAI chat should succeed");

    assert!(!result.response.is_empty(), "Should have a response");
    assert!(
        result.response.contains('4'),
        "Response should mention 4: {}",
        result.response
    );

    let captured = events.lock().unwrap();
    assert!(
        captured.iter().any(|e| e.contains("Token")),
        "Should have streamed tokens"
    );
}

#[tokio::test]
async fn test_openai_tool_calling() {
    let api_key = require_env!("OPENAI_API_KEY");
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    let provider = Arc::new(OpenAiProvider::new(
        "https://api.openai.com/v1",
        &api_key,
        &model,
    ));

    let result = run(
        provider,
        vec![
            ChatMessage::system("You are a calculator. Use the add tool to compute sums."),
            ChatMessage::user("What is 3 + 5?"),
        ],
        vec![Tool::function(
            "add",
            "Add two numbers",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }),
        )],
        |tc| {
            let args: serde_json::Value =
                parse_tool_args(&tc.function.arguments).unwrap_or_default();
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(format!("{}", a + b))
        },
        RunnerConfig::default(),
        CancellationToken::new(),
        Steering::new(),
        Guardrails::default(),
        |_| {},
    )
    .await
    .expect("Tool calling should succeed");

    assert!(
        result.response.contains('8'),
        "Response should mention 8: {}",
        result.response
    );
    assert!(result.total_usage.total_tokens > 0, "Should report usage");
}

#[tokio::test]
async fn test_anthropic_simple_chat() {
    let api_key = require_env!("ANTHROPIC_API_KEY");
    let model =
        std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());

    let provider = Arc::new(AnthropicProvider::new(&api_key, &model));

    let result = run(
        provider,
        vec![
            ChatMessage::system("Reply in one short sentence."),
            ChatMessage::user("What is the capital of France?"),
        ],
        vec![],
        |_| Ok("unused".into()),
        RunnerConfig {
            max_iterations: 1,
            ..Default::default()
        },
        CancellationToken::new(),
        Steering::new(),
        Guardrails::default(),
        |_| {},
    )
    .await
    .expect("Anthropic chat should succeed");

    assert!(!result.response.is_empty());
    assert!(
        result.response.to_lowercase().contains("paris"),
        "Response should mention Paris: {}",
        result.response
    );
}

#[tokio::test]
async fn test_azure_openai_chat() {
    let endpoint = require_env!("AZURE_OPENAI_ENDPOINT");
    let api_key = require_env!("AZURE_OPENAI_API_KEY");

    let provider = Arc::new(OpenAiProvider::new(&endpoint, &api_key, "unused"));

    let result = run(
        provider,
        vec![
            ChatMessage::system("Reply in one word."),
            ChatMessage::user("What color is the sky on a clear day?"),
        ],
        vec![],
        |_| Ok("unused".into()),
        RunnerConfig {
            max_iterations: 1,
            ..Default::default()
        },
        CancellationToken::new(),
        Steering::new(),
        Guardrails::default(),
        |_| {},
    )
    .await
    .expect("Azure OpenAI chat should succeed");

    assert!(!result.response.is_empty());
}
