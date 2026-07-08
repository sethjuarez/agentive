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

    let provider_clone = provider.clone();
    let cancel_clone = cancel.clone();
    let provider_handle =
        tokio::spawn(async move { provider_clone.chat(request, tx, &cancel_clone).await });

    let mut assistant_response: Option<ChatMessage> = None;

    while let Some(event) = rx.recv().await {
        match event {
            ChatEvent::Done { response } => {
                assistant_response = Some(response.message);
            }
            ChatEvent::Error { message } => {
                cancel.cancel();
                provider_handle.abort();
                return Err(AgentError::Stream(message));
            }
            _ => {}
        }
    }

    provider_handle
        .await
        .map_err(|e| AgentError::Stream(format!("Provider task panicked: {}", e)))??;

    assistant_response
        .ok_or_else(|| AgentError::Stream("No response received from provider".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;
    use std::time::Duration;

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

    #[tokio::test]
    async fn simple_chat_drains_streaming_tokens_while_provider_runs() {
        use crate::types::{ChatEvent, ChatRequest, ChatResponse};
        use tokio::sync::mpsc;

        struct StreamingProvider;

        #[async_trait::async_trait]
        impl Provider for StreamingProvider {
            async fn chat(
                &self,
                request: ChatRequest,
                tx: mpsc::Sender<ChatEvent>,
                _cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                assert!(!request.stream);

                for _ in 0..80 {
                    tx.send(ChatEvent::Token { token: "x".into() })
                        .await
                        .map_err(|e| AgentError::Stream(e.to_string()))?;
                }

                tx.send(ChatEvent::Done {
                    response: ChatResponse {
                        message: ChatMessage::assistant("finished"),
                        usage: None,
                    },
                })
                .await
                .map_err(|e| AgentError::Stream(e.to_string()))?;

                Ok(())
            }

            fn name(&self) -> &str {
                "streaming"
            }

            fn model(&self) -> Option<&str> {
                Some("streaming-model")
            }

            fn context_budget_chars(&self) -> usize {
                100_000
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(StreamingProvider);
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            simple_chat(provider, vec![ChatMessage::user("hello")]),
        )
        .await
        .expect("simple_chat should drain tokens concurrently")
        .unwrap();

        assert_eq!(result.text(), Some("finished"));
    }

    #[tokio::test]
    async fn simple_chat_returns_provider_error_when_no_done_event_arrives() {
        use crate::types::ChatRequest;
        use tokio::sync::mpsc;

        struct FailingProvider;

        #[async_trait::async_trait]
        impl Provider for FailingProvider {
            async fn chat(
                &self,
                _request: ChatRequest,
                _tx: mpsc::Sender<ChatEvent>,
                _cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                Err(AgentError::Api {
                    status: 500,
                    message: "provider failed".into(),
                })
            }

            fn name(&self) -> &str {
                "failing"
            }

            fn model(&self) -> Option<&str> {
                Some("failing-model")
            }

            fn context_budget_chars(&self) -> usize {
                100_000
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(FailingProvider);
        let err = simple_chat(provider, vec![ChatMessage::user("hello")])
            .await
            .unwrap_err();

        match err {
            AgentError::Api {
                status: 500,
                message,
            } => {
                assert_eq!(message, "provider failed");
            }
            other => panic!("expected provider API error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn simple_chat_returns_stream_error_event() {
        use crate::types::ChatRequest;
        use tokio::sync::mpsc;

        struct StreamErrorProvider;

        #[async_trait::async_trait]
        impl Provider for StreamErrorProvider {
            async fn chat(
                &self,
                _request: ChatRequest,
                tx: mpsc::Sender<ChatEvent>,
                cancel: &CancellationToken,
            ) -> Result<(), AgentError> {
                tx.send(ChatEvent::Token { token: "x".into() })
                    .await
                    .map_err(|e| AgentError::Stream(e.to_string()))?;
                tx.send(ChatEvent::Error {
                    message: "stream failed".into(),
                })
                .await
                .map_err(|e| AgentError::Stream(e.to_string()))?;

                tokio::time::sleep(Duration::from_secs(60)).await;
                assert!(cancel.is_cancelled());
                Ok(())
            }

            fn name(&self) -> &str {
                "stream-error"
            }

            fn model(&self) -> Option<&str> {
                Some("stream-error-model")
            }

            fn context_budget_chars(&self) -> usize {
                100_000
            }
        }

        let provider: Arc<dyn Provider> = Arc::new(StreamErrorProvider);
        let err = simple_chat(provider, vec![ChatMessage::user("hello")])
            .await
            .unwrap_err();

        match err {
            AgentError::Stream(message) => assert_eq!(message, "stream failed"),
            other => panic!("expected stream error, got {other:?}"),
        }
    }
}
