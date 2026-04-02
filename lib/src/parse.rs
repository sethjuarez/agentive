//! Robust JSON argument parsing for LLM-generated tool call arguments.
//!
//! LLMs sometimes produce malformed JSON — trailing commas, unescaped newlines,
//! markdown fences around JSON, etc. This module provides [`parse_tool_args`]
//! which tries progressively looser strategies before giving up.
//!
//! # Example
//! ```
//! use agentive::parse_tool_args;
//!
//! // Normal JSON works fine
//! let v = parse_tool_args(r#"{"path": "src/main.rs"}"#).unwrap();
//! assert_eq!(v["path"], "src/main.rs");
//!
//! // Tolerates markdown code fences
//! let v = parse_tool_args("```json\n{\"x\": 1}\n```").unwrap();
//! assert_eq!(v["x"], 1);
//! ```

/// Parse a tool call's `arguments` string into a [`serde_json::Value`].
///
/// Strategies (tried in order):
/// 1. Direct `serde_json::from_str`
/// 2. Strip markdown code fences and retry
/// 3. Extract first `{...}` block (greedy brace matching)
/// 4. Strip trailing commas before `}` or `]` and retry
pub fn parse_tool_args(raw: &str) -> Result<serde_json::Value, String> {
    let trimmed = raw.trim();

    // Strategy 1: direct parse
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return Ok(v);
    }

    // Strategy 2: strip markdown code fences
    let stripped = strip_code_fences(trimmed);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&stripped) {
        return Ok(v);
    }

    // Strategy 3: extract first {…} block with brace matching
    if let Some(block) = extract_json_block(&stripped) {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&block) {
            return Ok(v);
        }

        // Strategy 4: strip trailing commas and retry
        let cleaned = strip_trailing_commas(&block);
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned) {
            return Ok(v);
        }
    }

    // Strategy 4 on the full stripped input
    let cleaned = strip_trailing_commas(&stripped);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned) {
        return Ok(v);
    }

    Err(format!("Failed to parse tool arguments: {}", truncate(trimmed, 200)))
}

fn strip_code_fences(s: &str) -> String {
    let mut lines: Vec<&str> = s.lines().collect();

    // Remove leading fence (```json, ```JSON, ```, etc.)
    if let Some(first) = lines.first() {
        let ft = first.trim();
        if ft.starts_with("```") {
            lines.remove(0);
        }
    }

    // Remove trailing fence
    if let Some(last) = lines.last() {
        let lt = last.trim();
        if lt == "```" {
            lines.pop();
        }
    }

    lines.join("\n")
}

fn extract_json_block(s: &str) -> Option<String> {
    let start = s.find('{')?;
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for i in start..bytes.len() {
        let ch = bytes[i] as char;
        if escape {
            escape = false;
            continue;
        }
        if ch == '\\' && in_string {
            escape = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(s[start..=i].to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn strip_trailing_commas(s: &str) -> String {
    // Replace ,} with } and ,] with ]
    let mut result = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ',' {
            // Look ahead past whitespace for } or ]
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == '}' || chars[j] == ']') {
                // Skip the comma, keep the whitespace and closing bracket
                i += 1;
                continue;
            }
        }
        result.push(chars[i]);
        i += 1;
    }
    result
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        &s[..max]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_json() {
        let v = parse_tool_args(r#"{"path": "src/main.rs", "line": 42}"#).unwrap();
        assert_eq!(v["path"], "src/main.rs");
        assert_eq!(v["line"], 42);
    }

    #[test]
    fn test_markdown_fenced() {
        let input = "```json\n{\"query\": \"SELECT 1\"}\n```";
        let v = parse_tool_args(input).unwrap();
        assert_eq!(v["query"], "SELECT 1");
    }

    #[test]
    fn test_trailing_comma() {
        let input = r#"{"a": 1, "b": 2,}"#;
        let v = parse_tool_args(input).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], 2);
    }

    #[test]
    fn test_trailing_comma_in_array() {
        let input = r#"{"items": [1, 2, 3,]}"#;
        let v = parse_tool_args(input).unwrap();
        assert_eq!(v["items"][2], 3);
    }

    #[test]
    fn test_json_buried_in_text() {
        let input = "Here is the result:\n{\"answer\": 42}\nDone.";
        let v = parse_tool_args(input).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn test_empty_object() {
        let v = parse_tool_args("{}").unwrap();
        assert!(v.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_whitespace_wrapped() {
        let v = parse_tool_args("  \n  {\"x\": true}  \n  ").unwrap();
        assert_eq!(v["x"], true);
    }

    #[test]
    fn test_invalid_json_returns_error() {
        let result = parse_tool_args("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_braces() {
        let input = r#"{"config": {"nested": {"deep": true}}}"#;
        let v = parse_tool_args(input).unwrap();
        assert_eq!(v["config"]["nested"]["deep"], true);
    }

    #[test]
    fn test_string_with_braces() {
        let input = r#"{"code": "fn main() { println!(\"hello\"); }"}"#;
        let v = parse_tool_args(input).unwrap();
        assert!(v["code"].as_str().unwrap().contains("fn main"));
    }
}
