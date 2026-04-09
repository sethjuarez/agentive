//! Fetch URLs and extract clean text content from HTML.
//!
//! Provides `fetch_and_clean` for retrieving web pages and converting them
//! to readable text suitable for LLM context. Strips navigation, scripts,
//! styles, and other noise — keeping only the main article content.

use scraper::{Html, Selector};

const DEFAULT_MAX_CHARS: usize = 15_000;

/// Fetch a URL and return cleaned text content.
///
/// - For HTML responses: extracts main article content, strips noise elements
/// - For non-HTML responses (JSON, plain text): returns raw body
/// - Output is truncated to `max_chars` (default 15,000) to fit LLM context
pub async fn fetch_and_clean(url: &str) -> Result<String, String> {
    fetch_and_clean_with_limit(url, DEFAULT_MAX_CHARS).await
}

/// Fetch a URL and return cleaned text content with a custom character limit.
pub async fn fetch_and_clean_with_limit(url: &str, max_chars: usize) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    let resp = client
        .get(url)
        .header("User-Agent", "agentive/0.1 (Rust)")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch URL: {e}"))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(format!("HTTP {status} for {url}"));
    }

    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let body = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    if !content_type.contains("html") {
        return Ok(truncate(&body, max_chars));
    }

    Ok(truncate(&html_to_text(&body), max_chars))
}

/// Convert HTML to clean readable text, stripping scripts/styles/nav.
///
/// Tries to find main content areas (`<main>`, `<article>`, `[role=main]`,
/// `#content`, `.content`) first. Falls back to `<body>` with noise stripped.
pub fn html_to_text(html: &str) -> String {
    let doc = Html::parse_document(html);

    // Try to find main content areas first
    let main_selectors = ["main", "article", "[role=main]", "#content", ".content"];
    for sel_str in main_selectors {
        if let Ok(sel) = Selector::parse(sel_str) {
            let nodes: Vec<_> = doc.select(&sel).collect();
            if !nodes.is_empty() {
                let text: String = nodes
                    .iter()
                    .map(|n| extract_text(n))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                let cleaned = collapse_whitespace(&text);
                if cleaned.len() > 100 {
                    return cleaned;
                }
            }
        }
    }

    // Fallback: extract from body, skipping noise elements
    if let Ok(body_sel) = Selector::parse("body") {
        if let Some(body) = doc.select(&body_sel).next() {
            return collapse_whitespace(&extract_text(&body));
        }
    }

    // Last resort: all text
    collapse_whitespace(&doc.root_element().text().collect::<String>())
}

/// Recursively extract text from an element, skipping noise elements.
fn extract_text(el: &scraper::ElementRef) -> String {
    const SKIP_TAGS: &[&str] = &[
        "script", "style", "nav", "header", "footer", "noscript", "svg", "iframe",
    ];
    const BLOCK_TAGS: &[&str] = &[
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br",
        "blockquote", "pre", "section",
    ];

    let mut parts = Vec::new();
    for child in el.children() {
        match child.value() {
            scraper::node::Node::Text(t) => {
                let s = t.text.trim();
                if !s.is_empty() {
                    parts.push(s.to_string());
                }
            }
            scraper::node::Node::Element(e) => {
                let tag = e.name.local.as_ref();
                if SKIP_TAGS.contains(&tag) {
                    continue;
                }
                if let Some(child_el) = scraper::ElementRef::wrap(child) {
                    let child_text = extract_text(&child_el);
                    if !child_text.is_empty() {
                        if tag == "a" {
                            // Preserve links as markdown so agents can follow them
                            let href = e.attr("href").unwrap_or("");
                            if !href.is_empty()
                                && !href.starts_with('#')
                                && !href.starts_with("javascript:")
                            {
                                parts.push(format!("[{child_text}]({href})"));
                            } else {
                                parts.push(child_text);
                            }
                        } else if BLOCK_TAGS.contains(&tag) {
                            parts.push(format!("\n{child_text}\n"));
                        } else {
                            parts.push(child_text);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    parts.join(" ")
}

/// Collapse consecutive whitespace/newlines into clean text.
fn collapse_whitespace(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut prev_newline = false;
    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !prev_newline {
                result.push('\n');
                prev_newline = true;
            }
        } else {
            result.push_str(trimmed);
            result.push('\n');
            prev_newline = false;
        }
    }
    result.trim().to_string()
}

/// Truncate text to a maximum character length.
/// Uses a char boundary to avoid splitting multi-byte UTF-8 characters.
pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut pos = max;
    while pos > 0 && !s.is_char_boundary(pos) {
        pos -= 1;
    }
    format!("{}…\n\n[Truncated at {} chars]", &s[..pos], pos)
}

/// Create an agentive `Tool` definition for `fetch_url`.
///
/// Apps can include this in their tool list to let the LLM fetch URLs
/// autonomously. Use `fetch_and_clean` in the tool executor to handle calls.
pub fn fetch_url_tool() -> crate::types::Tool {
    crate::types::Tool::function(
        "fetch_url",
        "Fetch a web page and return its content as clean text. Use this to read documentation, articles, or any web resource. The HTML is automatically cleaned to extract the main content.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must start with http:// or https://)"
                }
            },
            "required": ["url"]
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_ascii() {
        let s = "abcdefghij";
        assert_eq!(truncate(s, 100), s);
        let t = truncate(s, 5);
        assert!(t.starts_with("abcde"));
        assert!(t.contains("[Truncated"));
    }

    #[test]
    fn truncate_multibyte_boundary() {
        let s = "aaaa🦀bbbb";
        let t = truncate(s, 5);
        assert!(t.starts_with("aaaa"));
        assert!(t.contains("[Truncated"));
    }

    #[test]
    fn truncate_all_multibyte() {
        let s = "🦀🦀🦀";
        let t = truncate(s, 5);
        assert!(t.starts_with("🦀"));
        assert!(t.contains("[Truncated"));
    }

    #[test]
    fn collapse_whitespace_normalizes() {
        let input = "  hello  \n\n\n  world  \n\n  end  ";
        let result = collapse_whitespace(input);
        assert_eq!(result, "hello\n\nworld\n\nend");
    }

    #[test]
    fn html_to_text_extracts_main() {
        let html = r#"
            <html><body>
                <nav>Menu Item 1</nav>
                <main><h1>Title</h1><p>Article content here.</p></main>
                <footer>Copyright</footer>
            </body></html>
        "#;
        let text = html_to_text(html);
        assert!(text.contains("Title"));
        assert!(text.contains("Article content here"));
        assert!(!text.contains("Menu Item"));
        assert!(!text.contains("Copyright"));
    }

    #[test]
    fn html_to_text_strips_scripts() {
        let html = r#"
            <html><body>
                <script>var x = 1;</script>
                <style>.foo { color: red; }</style>
                <p>Visible content</p>
            </body></html>
        "#;
        let text = html_to_text(html);
        assert!(text.contains("Visible content"));
        assert!(!text.contains("var x"));
        assert!(!text.contains("color: red"));
    }

    #[test]
    fn html_to_text_fallback_to_body() {
        let html = "<html><body><p>Just a paragraph.</p></body></html>";
        let text = html_to_text(html);
        assert!(text.contains("Just a paragraph"));
    }

    #[test]
    fn fetch_url_tool_definition() {
        let tool = fetch_url_tool();
        assert_eq!(tool.function.name, "fetch_url");
        let params = &tool.function.parameters;
        assert_eq!(params["required"][0], "url");
    }

    #[test]
    fn html_to_text_preserves_links_as_markdown() {
        let html = r#"
            <html><body>
                <main>
                    <p>Read the <a href="https://example.com/docs">documentation</a> for details.</p>
                    <p>Also see the <a href="/api/reference">API reference</a>.</p>
                </main>
            </body></html>
        "#;
        let text = html_to_text(html);
        assert!(text.contains("[documentation](https://example.com/docs)"));
        assert!(text.contains("[API reference](/api/reference)"));
    }

    #[test]
    fn html_to_text_skips_anchor_and_javascript_links() {
        let html = concat!(
            "<html><body><main>",
            r#"<p><a href=""#,
            r#"section">Jump to section</a></p>"#,
            r#"<p><a href="javascript:void(0)">Do nothing</a></p>"#,
            r#"<p><a href="">Empty href</a></p>"#,
            r#"<p><a>No href at all</a></p>"#,
            "</main></body></html>",
        );
        let text = html_to_text(html);
        // These should render as plain text, not markdown links
        assert!(text.contains("Jump to section"));
        assert!(!text.contains("](#"));
        assert!(!text.contains("javascript:"));
        assert!(text.contains("Do nothing"));
        assert!(text.contains("Empty href"));
        assert!(text.contains("No href at all"));
    }

    #[test]
    fn html_to_text_nested_link_content() {
        let html = r#"
            <html><body>
                <main>
                    <p>See <a href="https://example.com"><strong>bold link</strong> text</a> here.</p>
                </main>
            </body></html>
        "#;
        let text = html_to_text(html);
        assert!(text.contains("[bold link text](https://example.com)"));
    }
}
