/// Shared SSE (Server-Sent Events) line parser.
///
/// Buffers incoming byte chunks and yields complete SSE data lines.
pub struct SseParser {
    buffer: String,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
        }
    }

    /// Feed a chunk of bytes into the parser and extract any complete
    /// `data: ...` lines. Returns the data payloads (without the `data: ` prefix).
    /// Skips empty lines, comments (`:` prefix), and `event:` lines.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.push_str(&String::from_utf8_lossy(chunk));
        let mut results = Vec::new();

        while let Some(line_end) = self.buffer.find('\n') {
            let line = self.buffer[..line_end].trim().to_string();
            self.buffer = self.buffer[line_end + 1..].to_string();

            if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                results.push(data.trim().to_string());
            }
        }

        results
    }
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sse_parsing() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"data: {\"text\":\"hello\"}\n\ndata: {\"text\":\"world\"}\n\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "{\"text\":\"hello\"}");
        assert_eq!(lines[1], "{\"text\":\"world\"}");
    }

    #[test]
    fn test_partial_chunks() {
        let mut parser = SseParser::new();
        let lines1 = parser.feed(b"data: {\"te");
        assert!(lines1.is_empty());

        let lines2 = parser.feed(b"xt\":\"hello\"}\n\n");
        assert_eq!(lines2.len(), 1);
        assert_eq!(lines2[0], "{\"text\":\"hello\"}");
    }

    #[test]
    fn test_skips_comments_and_events() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b": comment\nevent: message\ndata: payload\n\n");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "payload");
    }

    #[test]
    fn test_done_signal() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"data: [DONE]\n\n");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "[DONE]");
    }
}
