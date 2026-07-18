//! Sanitization utilities for cleaning tool results before sending to APIs.
//!
//! Tool results can contain control characters, null bytes, and large
//! inline base64 data URIs. These cause JSON parse errors or bloat API
//! requests. Use [`sanitize_for_api`] to clean results before including
//! them in conversation history.

/// Strip control characters (except newlines, carriage returns, tabs) and
/// remove large inline base64 data URIs that bloat API requests without
/// adding LLM-readable value.
pub fn sanitize_for_api(s: &str) -> String {
    let stripped = strip_inline_base64(s);
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}

/// Replace inline `data:image/...;base64,...` URIs with a short placeholder.
/// These appear in markdown notes (pasted screenshots) and can bloat the body
/// by hundreds of KB without providing value to the LLM.
fn strip_inline_base64(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut remaining = s;

    while let Some(start) = remaining.find("data:image/") {
        result.push_str(&remaining[..start]);
        let after = &remaining[start..];

        if let Some(b64_start) = after.find(";base64,") {
            let data_start = b64_start + 8; // skip ";base64,"
            let data_end = after[data_start..]
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '+' && c != '/' && c != '=')
                .unwrap_or(after.len() - data_start);

            if data_end > 100 {
                result.push_str("[base64 image removed]");
            } else {
                result.push_str(&after[..data_start + data_end]);
            }
            remaining = &after[data_start + data_end..];
        } else {
            result.push_str("data:image/");
            remaining = &after[11..];
        }
    }

    result.push_str(remaining);
    result
}

/// Sanitize all text in a [`crate::types::ChatMessage`] — content, content parts, and
/// tool call arguments. Call this before sending messages to an LLM API
/// to prevent JSON parse errors from control characters in user input,
/// tool results, or model-generated content.
pub fn sanitize_message(msg: &mut crate::types::ChatMessage) {
    use crate::types::{ContentPart, MessageContent};

    if let Some(content) = &mut msg.content {
        match content {
            MessageContent::Text(t) => *t = sanitize_for_api(t),
            MessageContent::Parts(parts) => {
                for p in parts {
                    if let ContentPart::Text { text } = p {
                        *text = sanitize_for_api(text);
                    }
                }
            }
        }
    }
    if let Some(tool_calls) = &mut msg.tool_calls {
        for tc in tool_calls {
            tc.function.arguments = sanitize_for_api(&tc.function.arguments);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_strips_control_chars() {
        let input = "hello\x00world\x01!\nkeep newlines\ttabs too";
        let result = sanitize_for_api(input);
        assert_eq!(result, "helloworld!\nkeep newlines\ttabs too");
    }

    #[test]
    fn test_strip_large_base64() {
        let large_b64 = "A".repeat(200);
        let input = format!("before data:image/png;base64,{} after", large_b64);
        let result = sanitize_for_api(&input);
        assert_eq!(result, "before [base64 image removed] after");
    }

    #[test]
    fn test_keep_small_base64() {
        let input = "icon: data:image/png;base64,iVBOR end";
        let result = sanitize_for_api(input);
        assert!(result.contains("iVBOR"));
    }

    #[test]
    fn test_no_base64_passthrough() {
        let input = "normal text with no images";
        assert_eq!(sanitize_for_api(input), input);
    }

    #[test]
    fn test_sanitize_message_cleans_all_fields() {
        use crate::types::{ChatMessage, ContentPart, FunctionCall, MessageContent, ToolCall};

        let mut msg = ChatMessage {
            role: "assistant".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: "hello\x00world".into() },
            ])),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "test".into(),
                    arguments: "{\x01\"key\": \"val\"}".into(),
                },
            }]),
            tool_call_id: None,
        };

        sanitize_message(&mut msg);

        match msg.content.as_ref().unwrap() {
            MessageContent::Parts(parts) => {
                if let ContentPart::Text { text } = &parts[0] {
                    assert_eq!(text, "helloworld");
                } else {
                    panic!("Expected text part");
                }
            }
            _ => panic!("Expected parts"),
        }
        assert_eq!(
            msg.tool_calls.as_ref().unwrap()[0].function.arguments,
            "{\"key\": \"val\"}"
        );
    }
}
