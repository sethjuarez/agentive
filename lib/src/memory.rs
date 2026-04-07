//! Agent memory — persistent knowledge across conversations.
//!
//! A five-layer memory model inspired by human cognition:
//!
//! | Layer | Purpose | Managed by |
//! |-------|---------|-----------|
//! | **Working** | Active conversation context | `run()` + context trimming |
//! | **Procedural** | Tool definitions & system prompts | Static per session |
//! | **Core** | Persistent project/user facts | `save_memory` tool |
//! | **Archival** | Compressed session summaries | Auto-saved on session end |
//! | **Recall** | On-demand memory search | `recall_memory` tool |
//!
//! This module provides the in-memory data model, keyword search, system prompt
//! formatting, and standard tool definitions. **Persistence is pluggable** — apps
//! implement [`MemoryBackend`] to choose their storage (file, SQLite, cloud, etc.).
//!
//! # Example
//!
//! ```
//! use agentive::memory::{MemoryStore, MemoryCategory};
//!
//! let mut store = MemoryStore::default();
//! store.save(MemoryCategory::Core, "User prefers concise output", vec!["preference".into()]);
//! store.save(MemoryCategory::Insight, "Dashboard needs chart builder", vec!["dashboard".into()]);
//!
//! let results = store.recall("dashboard chart");
//! assert_eq!(results.len(), 1);
//! assert!(results[0].content.contains("Dashboard"));
//!
//! let prompt = store.format_for_system_prompt();
//! assert!(prompt.contains("concise output"));
//! ```

use crate::types::Tool;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

/// A single memory entry with category, content, tags, and timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Which memory category this belongs to.
    pub category: MemoryCategory,
    /// The content of the memory.
    pub content: String,
    /// When this memory was created (RFC 3339).
    pub created_at: String,
    /// Optional tags for search boosting and deduplication.
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Memory categories that determine how entries are stored and prioritized.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    /// Persistent facts about the user or project.
    /// Injected into every system prompt. Deduplicated by tags.
    Core,
    /// Compressed session summaries. Auto-evicted when cap is reached.
    Archival,
    /// Explicitly saved insights from conversations.
    Insight,
}

/// The full in-memory store. Serialize/deserialize for persistence backends.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryStore {
    /// All memory entries across categories.
    pub memories: Vec<MemoryEntry>,
}

/// Maximum number of memories before eviction kicks in.
const MAX_MEMORIES: usize = 200;

impl MemoryStore {
    /// Save a memory entry with deduplication and capacity management.
    ///
    /// For `Core` memories, entries with matching tags are replaced (dedup).
    /// When the store exceeds [`MAX_MEMORIES`], the oldest archival entries
    /// are evicted first.
    pub fn save(
        &mut self,
        category: MemoryCategory,
        content: &str,
        tags: Vec<String>,
    ) {
        let now = chrono::Utc::now().to_rfc3339();

        // Core memories: replace existing entry with same tags (dedup)
        if category == MemoryCategory::Core && !tags.is_empty() {
            self.memories.retain(|m| {
                !(m.category == MemoryCategory::Core && m.tags == tags)
            });
        }

        self.memories.push(MemoryEntry {
            category,
            content: content.to_string(),
            created_at: now,
            tags,
        });

        // Evict oldest archival entries when over capacity
        while self.memories.len() > MAX_MEMORIES {
            let archival_idx = self.memories.iter().position(|m| m.category == MemoryCategory::Archival);
            match archival_idx {
                Some(idx) => { self.memories.remove(idx); }
                None => break, // No archival to evict — hard stop
            }
        }
    }

    /// Save a session summary as an archival memory.
    pub fn archive_session(&mut self, summary: &str, session_id: &str) {
        self.save(
            MemoryCategory::Archival,
            summary,
            vec![format!("session:{session_id}")],
        );
    }

    /// Search memories by keyword. Returns up to 10 results sorted by relevance.
    ///
    /// Scoring: +2 per keyword match in content, +3 per keyword match in tags,
    /// +1 boost for core memories (only when already matched).
    pub fn recall(&self, query: &str) -> Vec<&MemoryEntry> {
        let query_lower = query.to_lowercase();
        let keywords: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(usize, &MemoryEntry)> = self.memories
            .iter()
            .filter_map(|m| {
                let content_lower = m.content.to_lowercase();
                let tag_text: String = m.tags.join(" ").to_lowercase();

                let mut score = 0usize;
                for kw in &keywords {
                    if content_lower.contains(kw) { score += 2; }
                    if tag_text.contains(kw) { score += 3; }
                }

                if score > 0 && m.category == MemoryCategory::Core {
                    score += 1;
                }

                if score > 0 { Some((score, m)) } else { None }
            })
            .collect();

        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored.into_iter().map(|(_, m)| m).take(10).collect()
    }

    /// Get all core memories (for system prompt injection).
    pub fn core_memories(&self) -> Vec<&MemoryEntry> {
        self.memories.iter()
            .filter(|m| m.category == MemoryCategory::Core)
            .collect()
    }

    /// Format core memories as a block for injection into the system prompt.
    /// Returns empty string if there are no core memories.
    pub fn format_for_system_prompt(&self) -> String {
        let cores = self.core_memories();
        if cores.is_empty() {
            return String::new();
        }
        let mut out = String::from("\n[Memories about this project and user]\n");
        for m in &cores {
            out.push_str(&format!("• {}\n", m.content));
        }
        out
    }

    /// Format recall results for LLM consumption.
    pub fn format_recall_results(results: &[&MemoryEntry]) -> String {
        if results.is_empty() {
            return "No memories found matching that query.".to_string();
        }
        let mut out = String::new();
        for (i, m) in results.iter().enumerate() {
            let cat = match m.category {
                MemoryCategory::Core => "core",
                MemoryCategory::Archival => "archival",
                MemoryCategory::Insight => "insight",
            };
            out.push_str(&format!("{}. [{}] {}\n", i + 1, cat, m.content));
            if !m.tags.is_empty() {
                out.push_str(&format!("   tags: {}\n", m.tags.join(", ")));
            }
        }
        out
    }

    /// Delete a memory by index. Returns the removed entry.
    pub fn delete(&mut self, index: usize) -> Option<MemoryEntry> {
        if index < self.memories.len() {
            Some(self.memories.remove(index))
        } else {
            None
        }
    }

    /// Update a memory's content by index. Returns `true` if the index was valid.
    pub fn update(&mut self, index: usize, content: &str) -> bool {
        if let Some(entry) = self.memories.get_mut(index) {
            entry.content = content.to_string();
            true
        } else {
            false
        }
    }

    /// Delete all memories of a given category, or all memories if `None`.
    /// Returns the number of entries removed.
    pub fn clear(&mut self, category: Option<MemoryCategory>) -> usize {
        let before = self.memories.len();
        match category {
            Some(cat) => self.memories.retain(|m| m.category != cat),
            None => self.memories.clear(),
        }
        before - self.memories.len()
    }
}

// ---------------------------------------------------------------------------
// Persistence trait
// ---------------------------------------------------------------------------

/// Backend for persisting memory stores.
///
/// Apps implement this trait to choose their storage strategy. The simplest
/// implementation writes JSON to a file; others might use SQLite, a database,
/// or a cloud service.
///
/// # Example
///
/// ```no_run
/// use agentive::memory::{MemoryBackend, MemoryStore};
///
/// struct FileBackend { path: std::path::PathBuf }
///
/// impl MemoryBackend for FileBackend {
///     fn load(&self) -> MemoryStore {
///         std::fs::read_to_string(&self.path)
///             .ok()
///             .and_then(|data| serde_json::from_str(&data).ok())
///             .unwrap_or_default()
///     }
///     fn save(&self, store: &MemoryStore) -> Result<(), String> {
///         let json = serde_json::to_string_pretty(store).map_err(|e| e.to_string())?;
///         std::fs::write(&self.path, json).map_err(|e| e.to_string())
///     }
/// }
/// ```
pub trait MemoryBackend: Send + Sync {
    /// Load the memory store. Returns empty store if nothing persisted.
    fn load(&self) -> MemoryStore;
    /// Persist the memory store. Called after every mutation.
    fn save(&self, store: &MemoryStore) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Standard `recall_memory` tool definition for LLM function calling.
pub fn recall_memory_tool() -> Tool {
    Tool::function(
        "recall_memory",
        "Search your memory for information from past conversations, saved facts, or session summaries. Use this when the user references something from a previous discussion, or when you need context about prior decisions. The search uses keyword matching — be specific.",
        serde_json::json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for in memory (e.g. 'color preference' or 'dashboard demo')"
                }
            }
        }),
    )
}

/// Standard `save_memory` tool definition for LLM function calling.
pub fn save_memory_tool() -> Tool {
    Tool::function(
        "save_memory",
        "Save an important fact, decision, or preference to memory so you can recall it in future conversations. Use 'core' for persistent facts about the user or project (e.g. preferences, tech stack, team info). Use 'insight' for decisions or conclusions from the current conversation.",
        serde_json::json!({
            "type": "object",
            "required": ["category", "content"],
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["core", "insight"],
                    "description": "Memory type: 'core' = persistent project/user facts, 'insight' = conversation conclusions"
                },
                "content": {
                    "type": "string",
                    "description": "The fact or insight to remember"
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional tags for search and deduplication (e.g. ['preference', 'narration-style'])"
                }
            }
        }),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_recall() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "User prefers TypeScript", vec!["language".into()]);
        store.save(MemoryCategory::Insight, "Dashboard needs chart builder", vec!["dashboard".into()]);
        store.save(MemoryCategory::Archival, "Session discussed login flow", vec!["session:1".into()]);

        let results = store.recall("dashboard chart");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("Dashboard"));

        let results = store.recall("TypeScript language");
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("TypeScript"));
    }

    #[test]
    fn core_dedup_by_tags() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "User likes blue", vec!["color-pref".into()]);
        store.save(MemoryCategory::Core, "User likes purple", vec!["color-pref".into()]);

        let cores = store.core_memories();
        assert_eq!(cores.len(), 1, "Should dedup core memories with same tags");
        assert!(cores[0].content.contains("purple"), "Should keep the latest value");
    }

    #[test]
    fn core_no_dedup_without_tags() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "Fact A", vec![]);
        store.save(MemoryCategory::Core, "Fact B", vec![]);

        assert_eq!(store.core_memories().len(), 2, "Empty tags should not trigger dedup");
    }

    #[test]
    fn archival_eviction_at_cap() {
        let mut store = MemoryStore::default();
        // Fill with 200 archival entries
        for i in 0..200 {
            store.save(MemoryCategory::Archival, &format!("session {i}"), vec![]);
        }
        assert_eq!(store.memories.len(), 200);

        // Add one core — should evict oldest archival
        store.save(MemoryCategory::Core, "important fact", vec!["key".into()]);
        assert_eq!(store.memories.len(), 200);
        // First entry should no longer be "session 0"
        assert!(!store.memories[0].content.contains("session 0"));
        // Last entry should be the core
        assert!(store.memories.last().unwrap().content.contains("important fact"));
    }

    #[test]
    fn recall_scores_tags_higher() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Insight, "We discussed the dashboard layout", vec![]);
        store.save(MemoryCategory::Core, "Monitoring setup", vec!["dashboard".into()]);

        let results = store.recall("dashboard");
        assert_eq!(results.len(), 2);
        // Tag match (3) + core boost (1) = 4 should beat content match (2)
        assert!(results[0].content.contains("Monitoring"));
    }

    #[test]
    fn recall_empty_query() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "Something", vec![]);
        assert!(store.recall("").is_empty(), "Empty query matches nothing");
    }

    #[test]
    fn format_system_prompt_empty() {
        let store = MemoryStore::default();
        assert!(store.format_for_system_prompt().is_empty());
    }

    #[test]
    fn format_system_prompt_with_cores() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "User prefers concise output", vec![]);
        store.save(MemoryCategory::Archival, "Old session", vec![]); // Not included

        let prompt = store.format_for_system_prompt();
        assert!(prompt.contains("concise output"));
        assert!(!prompt.contains("Old session"));
    }

    #[test]
    fn format_recall_results_output() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "Tech stack: Rust + React", vec!["tech".into()]);
        let results = store.recall("tech");
        let formatted = MemoryStore::format_recall_results(&results);
        assert!(formatted.contains("[core]"));
        assert!(formatted.contains("Rust + React"));
        assert!(formatted.contains("tags: tech"));
    }

    #[test]
    fn format_recall_empty() {
        let formatted = MemoryStore::format_recall_results(&[]);
        assert!(formatted.contains("No memories found"));
    }

    #[test]
    fn archive_session() {
        let mut store = MemoryStore::default();
        store.archive_session("Discussed login flow", "chat-2026-01");
        assert_eq!(store.memories.len(), 1);
        assert_eq!(store.memories[0].category, MemoryCategory::Archival);
        assert!(store.memories[0].tags.contains(&"session:chat-2026-01".to_string()));
    }

    #[test]
    fn delete_and_update() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "Original", vec![]);
        store.save(MemoryCategory::Insight, "Second", vec![]);

        assert!(store.update(0, "Updated"));
        assert_eq!(store.memories[0].content, "Updated");
        assert!(!store.update(99, "Out of bounds"));

        let removed = store.delete(0);
        assert!(removed.is_some());
        assert_eq!(store.memories.len(), 1);
        assert!(store.delete(99).is_none());
    }

    #[test]
    fn clear_by_category() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "Keep", vec![]);
        store.save(MemoryCategory::Archival, "Drop 1", vec![]);
        store.save(MemoryCategory::Archival, "Drop 2", vec![]);

        let removed = store.clear(Some(MemoryCategory::Archival));
        assert_eq!(removed, 2);
        assert_eq!(store.memories.len(), 1);
        assert_eq!(store.memories[0].content, "Keep");
    }

    #[test]
    fn clear_all() {
        let mut store = MemoryStore::default();
        store.save(MemoryCategory::Core, "A", vec![]);
        store.save(MemoryCategory::Insight, "B", vec![]);

        let removed = store.clear(None);
        assert_eq!(removed, 2);
        assert!(store.memories.is_empty());
    }

    #[test]
    fn tool_definitions() {
        let recall = recall_memory_tool();
        assert_eq!(recall.function.name, "recall_memory");

        let save = save_memory_tool();
        assert_eq!(save.function.name, "save_memory");
    }
}
