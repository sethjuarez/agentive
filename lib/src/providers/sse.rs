/// Shared SSE (Server-Sent Events) line parser.
///
/// Buffers incoming byte chunks and yields complete SSE data lines.
/// Handles multi-chunk UTF-8 by buffering raw bytes and only decoding
/// complete lines.
pub struct SseParser {
    buffer: Vec<u8>,
}

impl SseParser {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
        }
    }

    /// Feed a chunk of bytes into the parser and extract any complete
    /// `data: ...` lines. Returns the data payloads (without the `data: ` prefix).
    /// Skips empty lines, comments (`:` prefix), and `event:` lines.
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut results = Vec::new();

        loop {
            // Find a newline in the buffer
            let newline_pos = match self.buffer.iter().position(|&b| b == b'\n') {
                Some(pos) => pos,
                None => break,
            };

            // Extract the line (excluding the newline)
            let line_bytes = self.buffer[..newline_pos].to_vec();
            self.buffer = self.buffer[newline_pos + 1..].to_vec();

            // Decode to string (lossy for safety, but correct if chunks align)
            let line = String::from_utf8_lossy(&line_bytes);
            let line = line.trim();

            if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
                continue;
            }

            if let Some(data) = line.strip_prefix("data: ") {
                results.push(data.trim().to_string());
            } else if let Some(data) = line.strip_prefix("data:") {
                // Handle "data:payload" without space after colon
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

    #[test]
    fn test_crlf_line_endings() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"data: payload1\r\ndata: payload2\r\n");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "payload1");
        assert_eq!(lines[1], "payload2");
    }

    #[test]
    fn test_multibyte_utf8_split_across_chunks() {
        let mut parser = SseParser::new();
        // 'é' is [0xC3, 0xA9] in UTF-8. Split it across two chunks.
        let chunk1: &[u8] = &[b'd', b'a', b't', b'a', b':', b' ', 0xC3];
        let chunk2: &[u8] = &[0xA9, b'\n'];
        let lines1 = parser.feed(chunk1);
        assert!(lines1.is_empty()); // no newline yet
        let lines2 = parser.feed(chunk2);
        assert_eq!(lines2.len(), 1);
        assert_eq!(lines2[0], "é");
    }

    #[test]
    fn test_emoji_in_sse_data() {
        let mut parser = SseParser::new();
        let lines = parser.feed("data: 🦀 Hello\n\n".as_bytes());
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "🦀 Hello");
    }

    #[test]
    fn test_realistic_openai_stream() {
        let mut parser = SseParser::new();
        // Simulates a realistic OpenAI streaming response split across chunks
        let chunk1 = b"data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\nda";
        let chunk2 = b"ta: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\ndata: [DONE]\n\n";

        let lines1 = parser.feed(chunk1);
        assert_eq!(lines1.len(), 1);
        assert!(lines1[0].contains("Hello"));

        let lines2 = parser.feed(chunk2);
        assert_eq!(lines2.len(), 2);
        assert!(lines2[0].contains("world"));
        assert_eq!(lines2[1], "[DONE]");
    }

    #[test]
    fn test_data_without_space_after_colon() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"data:no-space-payload\n");
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0], "no-space-payload");
    }

    #[test]
    fn test_interleaved_events_and_data() {
        let mut parser = SseParser::new();
        let input = b"event: message\ndata: first\n\nevent: tool\ndata: second\n\n";
        let lines = parser.feed(input);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "first");
        assert_eq!(lines[1], "second");
    }

    #[test]
    fn test_empty_stream() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"");
        assert!(lines.is_empty());
    }

    #[test]
    fn test_only_newlines() {
        let mut parser = SseParser::new();
        let lines = parser.feed(b"\n\n\n");
        assert!(lines.is_empty());
    }
}
