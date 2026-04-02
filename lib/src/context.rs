//! Context window management — trimming and summarization.
//!
//! When conversations grow too long for the model's context window,
//! this module trims older messages and produces summaries to preserve context.

use crate::types::ChatMessage;

/// Estimate the character cost of a slice of messages.
pub fn estimate_chars(msgs: &[ChatMessage]) -> usize {
    msgs.iter()
        .map(|m| {
            let content_len = m.content.as_ref().map_or(0, |c| c.char_len());
            let tool_len = m.tool_calls.as_ref().map_or(0, |tc| {
                tc.iter()
                    .map(|t| t.function.name.chars().count() + t.function.arguments.chars().count())
                    .sum()
            });
            let tool_id_len = m.tool_call_id.as_ref().map_or(0, |id| id.chars().count());
            content_len + tool_len + tool_id_len + 20 // overhead per message
        })
        .sum()
}

/// Truncate a string to at most `max` bytes on a valid UTF-8 boundary.
fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        &s[..end]
    }
}

/// Build a compact string summary from messages that are about to be dropped.
/// Extracts key user requests, assistant decisions, and tool actions without
/// needing an LLM call.
pub fn summarize_dropped(dropped: &[ChatMessage]) -> String {
    let mut parts: Vec<String> = Vec::new();

    for msg in dropped {
        match msg.role.as_str() {
            "user" => {
                if let Some(text) = msg.text() {
                    let truncated = truncate_str(text, 200);
                    parts.push(format!("• User asked: {truncated}"));
                }
            }
            "assistant" => {
                if let Some(text) = msg.text() {
                    let truncated = truncate_str(text, 200);
                    parts.push(format!("• Assistant: {truncated}"));
                }
                if let Some(tool_calls) = &msg.tool_calls {
                    for tc in tool_calls {
                        parts.push(format!("• Called tool: {}", tc.function.name));
                    }
                }
            }
            "tool" => {
                // Skip tool results — they're verbose and the tool call name is enough
            }
            _ => {}
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    // Cap the summary at ~4000 chars
    let mut result = String::from("[Earlier conversation summary]\n");
    for part in &parts {
        if result.len() + part.len() > 4000 {
            result.push_str("\n• ... (older messages omitted)");
            break;
        }
        result.push_str(part);
        result.push('\n');
    }
    result
}

/// Trim messages to fit within the character budget, preserving context
/// via a compact summary of dropped messages.
///
/// Strategy:
/// 1. Keep system messages at the front
/// 2. If over budget, drop the oldest non-system messages
/// 3. Summarize them into a "memory" user message
/// 4. Insert the summary after system messages, before recent conversation
///
/// Returns `(dropped_count, dropped_messages)` so the caller can optionally
/// upgrade the summary via LLM.
pub fn trim_to_context_window(
    messages: &mut Vec<ChatMessage>,
    max_chars: usize,
) -> (usize, Vec<ChatMessage>) {
    if estimate_chars(messages) <= max_chars {
        return (0, Vec::new());
    }

    // Split into system prefix and conversation
    let system_end = messages
        .iter()
        .position(|m| m.role != "system")
        .unwrap_or(messages.len());
    let system_msgs: Vec<ChatMessage> = messages.drain(..system_end).collect();
    let system_chars = estimate_chars(&system_msgs);

    // Reserve space for the summary message (~5k chars max)
    let budget = max_chars.saturating_sub(system_chars).saturating_sub(5000);

    // Drop messages from the front
    let mut dropped: Vec<ChatMessage> = Vec::new();
    while estimate_chars(messages) > budget && messages.len() > 2 {
        dropped.push(messages.remove(0));
    }

    let dropped_count = dropped.len();

    // Build fast string summary
    let summary = summarize_dropped(&dropped);

    // Reassemble: system + summary + recent conversation
    let recent = std::mem::take(messages);
    *messages = system_msgs;
    if !summary.is_empty() {
        messages.push(ChatMessage::user(&summary));
    }
    messages.extend(recent);

    (dropped_count, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_estimate_chars() {
        let msgs = vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there, how can I help?"),
        ];
        let chars = estimate_chars(&msgs);
        assert!(chars > 30); // "Hello" + "Hi there..." + overhead
    }

    #[test]
    fn test_summarize_dropped() {
        let msgs = vec![
            ChatMessage::user("Write a blog post about Rust"),
            ChatMessage::assistant("Sure, let me read some context first."),
            ChatMessage::assistant_with_tool_calls(vec![ToolCall {
                id: "call_1".into(),
                call_type: "function".into(),
                function: FunctionCall {
                    name: "read_file".into(),
                    arguments: "{}".into(),
                },
            }]),
            ChatMessage::tool_result("call_1", "file contents here"),
        ];

        let summary = summarize_dropped(&msgs);
        assert!(summary.contains("User asked:"));
        assert!(summary.contains("Called tool: read_file"));
        assert!(!summary.contains("file contents here")); // tool results excluded
    }

    #[test]
    fn test_trim_no_op_when_under_budget() {
        let mut msgs = vec![
            ChatMessage::system("You are helpful"),
            ChatMessage::user("Hi"),
        ];
        let (dropped, _) = trim_to_context_window(&mut msgs, 100_000);
        assert_eq!(dropped, 0);
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn test_trim_drops_oldest_messages() {
        let mut msgs = vec![ChatMessage::system("system prompt")];
        // Add many messages to exceed budget
        for i in 0..50 {
            msgs.push(ChatMessage::user(&format!("message {}: {}", i, "x".repeat(500))));
            msgs.push(ChatMessage::assistant(&format!("response {}: {}", i, "y".repeat(500))));
        }

        let total_before = msgs.len();
        let (dropped_count, _) = trim_to_context_window(&mut msgs, 5000);

        assert!(dropped_count > 0);
        assert!(msgs.len() < total_before);
        // System message should still be first
        assert_eq!(msgs[0].role, "system");
        // Summary should be inserted after system
        assert_eq!(msgs[1].role, "user");
        assert!(msgs[1].text().unwrap().contains("[Earlier conversation summary]"));
    }

    #[test]
    fn test_summarize_multibyte_utf8() {
        // Messages with CJK characters — should not panic on truncation
        let long_text = "日本語のテキスト".repeat(50); // well over 200 bytes
        let msgs = vec![
            ChatMessage::user(&long_text),
            ChatMessage::assistant(&long_text),
        ];
        let summary = summarize_dropped(&msgs);
        assert!(summary.contains("User asked:"));
        assert!(summary.contains("Assistant:"));
        // No panic = success
    }

    #[test]
    fn test_estimate_chars_multibyte() {
        // 'é' is 2 bytes but 1 char; char_len should count chars now
        let msgs = vec![ChatMessage::user("héllo")];
        let chars = estimate_chars(&msgs);
        // "héllo" = 5 chars + 20 overhead = 25
        assert_eq!(chars, 25);
    }

    #[test]
    fn test_trim_only_two_messages_over_budget() {
        // Edge case: 2 non-system messages that exceed budget
        let mut msgs = vec![
            ChatMessage::user(&"x".repeat(5000)),
            ChatMessage::assistant(&"y".repeat(5000)),
        ];
        // Budget smaller than content — trimming stops at 2 messages
        let (dropped, _) = trim_to_context_window(&mut msgs, 100);
        // Should not panic, may not be able to trim below budget
        assert_eq!(dropped, 0); // can't drop below 2
    }

    #[test]
    fn test_trim_empty_messages() {
        let mut msgs: Vec<ChatMessage> = Vec::new();
        let (dropped, _) = trim_to_context_window(&mut msgs, 100);
        assert_eq!(dropped, 0);
        assert!(msgs.is_empty());
    }
}
