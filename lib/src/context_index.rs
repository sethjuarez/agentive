//! Host-neutral context and recall index primitives.
//!
//! Agent loops often need durable, high-signal records of what context was made
//! available, which references were resolved, and which resources should be
//! searchable later. This module models those records without prescribing a
//! database, embedding strategy, or UI.

use crate::state::TouchedResource;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Origin for context made available to a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextSource {
    User,
    System,
    ToolResult,
    File,
    Search,
    Checkpoint,
    Memory,
    Host,
    Custom,
}

/// A reusable context item that can be recalled or injected into a later turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextItem {
    /// Stable host- or agent-generated ID.
    pub id: String,
    /// Source of the context.
    pub source: ContextSource,
    /// Human-friendly name.
    pub name: String,
    /// Short description for display and recall ranking.
    pub description: String,
    /// Optional content or artifact reference. Hosts may store only a summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// MIME type or host-defined content type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Number of times this item has been read/injected.
    #[serde(default)]
    pub read_count: u32,
    /// Optional host-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl ContextItem {
    /// Create a context item without embedded content.
    pub fn new(
        id: impl Into<String>,
        source: ContextSource,
        name: impl Into<String>,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source,
            name: name.into(),
            description: description.into(),
            content: None,
            content_type: None,
            read_count: 0,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach content and content type.
    pub fn with_content(
        mut self,
        content: impl Into<String>,
        content_type: impl Into<String>,
    ) -> Self {
        self.content = Some(content.into());
        self.content_type = Some(content_type.into());
        self
    }

    /// Attach metadata while preserving builder-style construction.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// A reference encountered or resolved during a run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReferenceRecord {
    /// Host-defined reference type, such as `file`, `url`, `symbol`, or `memory`.
    pub ref_type: String,
    /// Reference value as written or resolved.
    pub value: String,
    /// Turn index where the reference appeared, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    /// Whether the reference was successfully resolved.
    pub resolved: bool,
    /// Optional host-specific metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

impl ReferenceRecord {
    /// Create a reference record.
    pub fn new(ref_type: impl Into<String>, value: impl Into<String>, resolved: bool) -> Self {
        Self {
            ref_type: ref_type.into(),
            value: value.into(),
            turn_index: None,
            resolved,
            metadata: BTreeMap::new(),
        }
    }

    /// Attach the turn index.
    pub fn with_turn_index(mut self, turn_index: usize) -> Self {
        self.turn_index = Some(turn_index);
        self
    }
}

/// A resource entry suitable for a host-owned search or recall index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchIndexEntry {
    /// Resource identity and operation.
    pub resource: TouchedResource,
    /// Display title.
    pub title: String,
    /// Short searchable summary.
    pub summary: String,
    /// Optional content type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Optional tags for filtering or ranking.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl SearchIndexEntry {
    /// Create a search index entry.
    pub fn new(
        resource: TouchedResource,
        title: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            resource,
            title: title.into(),
            summary: summary.into(),
            content_type: None,
            tags: Vec::new(),
        }
    }

    /// Attach content type.
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    /// Attach a tag.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ResourceOperation, TouchedResource};

    #[test]
    fn serializes_context_item_shape() {
        let item = ContextItem::new(
            "ctx-1",
            ContextSource::Checkpoint,
            "Latest checkpoint",
            "High-signal resume state",
        )
        .with_content("Goal and next step", "text/plain")
        .with_metadata("session", "abc");

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains("\"source\":\"checkpoint\""));
        let round_trip: ContextItem = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, item);

        let custom = ContextItem::new(
            "ctx-2",
            ContextSource::Custom,
            "Host context",
            "Host-defined source",
        );
        let custom_json = serde_json::to_string(&custom).unwrap();
        assert!(custom_json.contains("\"source\":\"custom\""));
    }

    #[test]
    fn models_refs_and_search_entries() {
        let reference = ReferenceRecord::new("file", "lib/src/lib.rs", true).with_turn_index(3);
        assert_eq!(reference.turn_index, Some(3));

        let entry = SearchIndexEntry::new(
            TouchedResource::new("file", "lib/src/lib.rs", ResourceOperation::Read),
            "Public API exports",
            "Re-exports agentive modules",
        )
        .with_content_type("text/rust")
        .with_tag("api");

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"tags\":[\"api\"]"));
        let round_trip: SearchIndexEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(round_trip, entry);
    }
}
