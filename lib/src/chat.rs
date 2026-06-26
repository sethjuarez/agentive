//! One-shot non-streaming chat helper.
//!
//! Many apps need a simple "send messages, get one response" without the full
//! agentic loop (no tools, no streaming, no retries).  `simple_chat` wraps a
//! provider's `chat()` method with the boilerplate: channel setup, request
//! construction, and response extraction.
//!
//! # Example
//! ```no_run
//! use agentive::{simple_chat, ChatMessage};
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), agentive::AgentError> {
//! let provider = agentive::build_provider(
//!     "https://my-resource.openai.azure.com",
//!     "my-api-key",
//!     "gpt-4o",
//! );
//!
//! let response = simple_chat(
//!     provider,
//!     vec![
//!         ChatMessage::system("You are a helpful assistant."),
//!         ChatMessage::user("What is 2 + 2?"),
//!     ],
//! ).await?;
//!
//! println!("{}", response.text().unwrap_or("no response"));
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use crate::cancel::CancellationToken;
use crate::error::AgentError;
use crate::provider::Provider;
use crate::types::{ChatEvent, ChatMessage, ChatRequest};

/// Perform a single non-streaming chat turn.
///
/// Sends the messages to the provider and returns the assistant's response
/// message.  No tools, no streaming, no retries — just a clean request/response.
///
/// Use this for quick operations like text reformatting, summarisation,
/// or any one-shot LLM call that doesn't need the agentic loop.
pub async fn simple_chat(
    provider: Arc<dyn Provider>,
    messages: Vec<ChatMessage>,
) -> Result<ChatMessage, AgentError> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(64);
    let cancel = CancellationToken::new();
    let request = ChatRequest {
        messages,
        model: provider.model().unwrap_or_default().to_string(),
        tools: None,
        stream: false,
        response_format: None,
    };

    provider.chat(request, tx, &cancel).await?;

    while let Some(event) = rx.recv().await {
        match event {
            ChatEvent::Done { response } => return Ok(response.message),
            ChatEvent::Error { message } => {
                return Err(AgentError::Stream(message));
            }
            _ => {}
        }
    }

    Err(AgentError::Stream(
        "No response received from provider".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[tokio::test]
    async fn simple_chat_returns_assistant_message() {
        use crate::types::{ChatEvent, ChatRequest, ChatResponse, Usage};
        use tokio::sync::mpsc;

        struct EchoProvider;

        #[async_trait::async_trait]
        impl Provider for EchoProvider {
            async fn chat(
                &self,
                request: ChatRequest,
                tx: mpsc::Sender<ChatEvent>,
                _cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                assert_eq!(request.model, "echo-model");
                let user_text = request
                    .messages
                    .last()
                    .and_then(|m| m.text())
                    .unwrap_or("no input")
                    .to_string();
                let _ = tx
                    .send(ChatEvent::Done {
                        response: ChatResponse {
                            message: ChatMessage::assistant(&format!("Echo: {user_text}")),
                            usage: Some(Usage {
                                prompt_tokens: 10,
                                completion_tokens: 5,
                                total_tokens: 15,
                            }),
                        },
                    })
                    .await;
                Ok(())
            }

            fn name(&self) -> &str {
                "echo"
            }

            fn model(&self) -> Option<&str> {
                Some("echo-model")
            }

            fn context_budget_chars(&self) -> usize {
                100_000
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(EchoProvider);
        let result = simple_chat(provider, vec![ChatMessage::user("hello")])
            .await
            .unwrap();

        assert_eq!(result.text(), Some("Echo: hello"));
        assert_eq!(result.role, "assistant");
    }
}
