//! Host-neutral context and recall index primitives.
//!
//! Agent loops often need durable, high-signal records of what context was made
//! available, which references were resolved, and which resources should be
//! searchable later. This module models those records without prescribing a
//! database, embedding strategy, or UI.

use crate::state::TouchedResource;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

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

/// Semantic role for a context item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ContextKind {
    RecentTurn,
    MemoryFact,
    ReferenceDoc,
    ToolObservation,
    FileExcerpt,
    WebExcerpt,
    ErrorTrace,
    MediaSummary,
    Other,
}

impl Default for ContextKind {
    fn default() -> Self {
        Self::Other
    }
}

/// Sensitivity label used by packers before context is injected into a prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ContextSensitivity {
    Public,
    Internal,
    Private,
    Secret,
}

impl Default for ContextSensitivity {
    fn default() -> Self {
        Self::Internal
    }
}

/// Visibility scope for context recall and packing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ContextScope {
    Session,
    Project,
    User,
    Global,
}

impl Default for ContextScope {
    fn default() -> Self {
        Self::Session
    }
}

/// Reference to large context stored outside the prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct LargeContextRef {
    /// Stable payload ID in a host-owned store.
    pub id: String,
    /// Tool or resource name that can expand this reference.
    pub expand_tool: String,
    /// Optional byte size for display and policy decisions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<usize>,
    /// Optional content hash for deduplication.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl LargeContextRef {
    pub fn new(id: impl Into<String>, expand_tool: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            expand_tool: expand_tool.into(),
            bytes: None,
            hash: None,
        }
    }

    pub fn with_bytes(mut self, bytes: usize) -> Self {
        self.bytes = Some(bytes);
        self
    }

    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
        self
    }
}

/// A reusable context item that can be recalled or injected into a later turn.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextItem {
    /// Stable host- or agent-generated ID.
    pub id: String,
    /// Source of the context.
    pub source: ContextSource,
    /// Human-friendly name.
    pub name: String,
    /// Short description for display and recall ranking.
    pub description: String,
    /// Semantic role for budget allocation and prompt layout.
    #[serde(default)]
    pub kind: ContextKind,
    /// Priority where larger values are packed first when relevance ties.
    #[serde(default)]
    pub priority: i32,
    /// Visibility and privacy label.
    #[serde(default)]
    pub sensitivity: ContextSensitivity,
    /// Scope used by hosts when deciding whether context can cross sessions.
    #[serde(default)]
    pub scope: ContextScope,
    /// Optional content or artifact reference. Hosts may store only a summary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// MIME type or host-defined content type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Optional reference to the full payload outside the prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub large_ref: Option<LargeContextRef>,
    /// Optional host/model token estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_tokens: Option<usize>,
    /// Optional serialized byte estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimated_bytes: Option<usize>,
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
            kind: ContextKind::Other,
            priority: 0,
            sensitivity: ContextSensitivity::Internal,
            scope: ContextScope::Session,
            content: None,
            content_type: None,
            large_ref: None,
            estimated_tokens: None,
            estimated_bytes: None,
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

    /// Set semantic kind.
    pub fn with_kind(mut self, kind: ContextKind) -> Self {
        self.kind = kind;
        self
    }

    /// Set packing priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set sensitivity.
    pub fn with_sensitivity(mut self, sensitivity: ContextSensitivity) -> Self {
        self.sensitivity = sensitivity;
        self
    }

    /// Set visibility scope.
    pub fn with_scope(mut self, scope: ContextScope) -> Self {
        self.scope = scope;
        self
    }

    /// Attach a reference to full payload content stored outside the prompt.
    pub fn with_large_ref(mut self, large_ref: LargeContextRef) -> Self {
        self.large_ref = Some(large_ref);
        self
    }

    /// Attach host-provided estimates.
    pub fn with_estimates(mut self, tokens: Option<usize>, bytes: Option<usize>) -> Self {
        self.estimated_tokens = tokens;
        self.estimated_bytes = bytes;
        self
    }

    /// Attach metadata while preserving builder-style construction.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Estimate prompt bytes for this item using host estimate when available.
    pub fn estimated_prompt_bytes(&self) -> usize {
        self.estimated_bytes.unwrap_or_else(|| {
            self.name.len()
                + self.description.len()
                + self
                    .content
                    .as_ref()
                    .map(|content| content.len())
                    .unwrap_or(0)
                + 128
        })
    }

    fn searchable_text(&self) -> String {
        let mut text = format!("{} {}", self.name, self.description);
        if let Some(content) = &self.content {
            text.push(' ');
            text.push_str(content);
        }
        for (key, value) in &self.metadata {
            text.push(' ');
            text.push_str(key);
            text.push(' ');
            text.push_str(value);
        }
        text
    }
}

/// Per-kind budget override.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextKindBudget {
    pub kind: ContextKind,
    pub max_bytes: usize,
    pub max_items: usize,
}

impl ContextKindBudget {
    pub fn new(kind: ContextKind, max_bytes: usize, max_items: usize) -> Self {
        Self {
            kind,
            max_bytes,
            max_items,
        }
    }
}

/// Configuration for deterministic context packing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextPackingConfig {
    pub total_budget_bytes: usize,
    pub default_kind_budget_bytes: usize,
    pub default_kind_max_items: usize,
    pub max_item_preview_bytes: usize,
    pub include_private: bool,
    pub include_secret: bool,
    #[serde(default)]
    pub kind_budgets: Vec<ContextKindBudget>,
}

impl Default for ContextPackingConfig {
    fn default() -> Self {
        Self {
            total_budget_bytes: 24_000,
            default_kind_budget_bytes: 8_000,
            default_kind_max_items: 8,
            max_item_preview_bytes: 4_000,
            include_private: true,
            include_secret: false,
            kind_budgets: vec![
                ContextKindBudget::new(ContextKind::RecentTurn, 8_000, 8),
                ContextKindBudget::new(ContextKind::MemoryFact, 3_000, 10),
                ContextKindBudget::new(ContextKind::ReferenceDoc, 8_000, 6),
                ContextKindBudget::new(ContextKind::ToolObservation, 6_000, 6),
                ContextKindBudget::new(ContextKind::FileExcerpt, 8_000, 6),
                ContextKindBudget::new(ContextKind::WebExcerpt, 6_000, 4),
                ContextKindBudget::new(ContextKind::ErrorTrace, 6_000, 4),
                ContextKindBudget::new(ContextKind::MediaSummary, 3_000, 4),
            ],
        }
    }
}

impl ContextPackingConfig {
    fn budget_for(&self, kind: &ContextKind) -> (usize, usize) {
        self.kind_budgets
            .iter()
            .find(|budget| &budget.kind == kind)
            .map(|budget| (budget.max_bytes, budget.max_items))
            .unwrap_or((self.default_kind_budget_bytes, self.default_kind_max_items))
    }
}

/// Packer decision for observability and evals.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct ContextPackDecision {
    pub id: String,
    pub kind: ContextKind,
    pub action: ContextPackAction,
    pub reason: String,
    pub score: f32,
    pub estimated_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextPackAction {
    Selected,
    Previewed,
    Dropped,
    Redacted,
}

/// Item selected for injection into the prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct PackedContextItem {
    pub id: String,
    pub source: ContextSource,
    pub kind: ContextKind,
    pub name: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub large_ref: Option<LargeContextRef>,
    pub estimated_bytes: usize,
}

/// Result of packing context for a model call.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct PackedContext {
    pub items: Vec<PackedContextItem>,
    pub decisions: Vec<ContextPackDecision>,
    pub total_bytes: usize,
    pub budget_bytes: usize,
}

impl PackedContext {
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Render selected context as a stable, source-attributed prompt block.
    pub fn to_prompt_block(&self) -> String {
        if self.items.is_empty() {
            return String::new();
        }

        let mut out = String::from("<context_pack>\n");
        for item in &self.items {
            out.push_str(&render_packed_item(item));
        }
        out.push_str("</context_pack>");
        out
    }
}

/// Deterministic budgeted context packer.
pub struct ContextPacker;

impl ContextPacker {
    pub fn pack(
        query: &str,
        items: &[ContextItem],
        config: &ContextPackingConfig,
    ) -> PackedContext {
        let query_terms = tokenize(query);
        let mut candidates = items
            .iter()
            .map(|item| {
                let score = score_item(item, &query_terms);
                (item, score)
            })
            .collect::<Vec<_>>();

        candidates.sort_by(|(a, a_score), (b, b_score)| {
            b_score
                .partial_cmp(a_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.priority.cmp(&a.priority))
                .then_with(|| a.kind.cmp(&b.kind))
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut decisions = Vec::new();
        let mut packed = Vec::new();
        let wrapper_bytes = "<context_pack>\n</context_pack>".len();
        let item_budget_bytes = config.total_budget_bytes.saturating_sub(wrapper_bytes);
        let mut item_bytes = 0usize;
        let mut kind_bytes: BTreeMap<ContextKind, usize> = BTreeMap::new();
        let mut kind_counts: BTreeMap<ContextKind, usize> = BTreeMap::new();
        let mut seen_refs = BTreeSet::new();

        for (item, score) in candidates {
            let estimate = item.estimated_prompt_bytes();
            if should_redact(item, config) {
                decisions.push(decision(
                    item,
                    ContextPackAction::Redacted,
                    "sensitivity policy excluded item",
                    score,
                    estimate,
                ));
                continue;
            }

            if let Some(large_ref) = &item.large_ref {
                if seen_refs.contains(&large_ref.id) {
                    decisions.push(decision(
                        item,
                        ContextPackAction::Dropped,
                        "duplicate full-payload reference",
                        score,
                        estimate,
                    ));
                    continue;
                }
            }

            let (kind_budget, kind_max_items) = config.budget_for(&item.kind);
            let used_kind_bytes = *kind_bytes.get(&item.kind).unwrap_or(&0);
            let used_kind_items = *kind_counts.get(&item.kind).unwrap_or(&0);
            if used_kind_items >= kind_max_items {
                decisions.push(decision(
                    item,
                    ContextPackAction::Dropped,
                    "kind item budget exhausted",
                    score,
                    estimate,
                ));
                continue;
            }
            if item_bytes >= item_budget_bytes || used_kind_bytes >= kind_budget {
                decisions.push(decision(
                    item,
                    ContextPackAction::Dropped,
                    "byte budget exhausted",
                    score,
                    estimate,
                ));
                continue;
            }

            let remaining_total = item_budget_bytes - item_bytes;
            let remaining_kind = kind_budget - used_kind_bytes;
            let Some((content, action, packed_bytes)) = fit_packed_item(
                item,
                remaining_total.min(remaining_kind),
                config.max_item_preview_bytes,
            ) else {
                decisions.push(decision(
                    item,
                    ContextPackAction::Dropped,
                    "not enough remaining budget for useful preview",
                    score,
                    estimate,
                ));
                continue;
            };

            item_bytes += packed_bytes;
            *kind_bytes.entry(item.kind.clone()).or_default() += packed_bytes;
            *kind_counts.entry(item.kind.clone()).or_default() += 1;
            if let Some(large_ref) = &item.large_ref {
                seen_refs.insert(large_ref.id.clone());
            }
            decisions.push(decision(
                item,
                action,
                "selected for prompt context",
                score,
                packed_bytes,
            ));
            packed.push(PackedContextItem {
                id: item.id.clone(),
                source: item.source.clone(),
                kind: item.kind.clone(),
                name: item.name.clone(),
                content,
                content_type: item.content_type.clone(),
                large_ref: item.large_ref.clone(),
                estimated_bytes: packed_bytes,
            });
        }

        let total_bytes = if packed.is_empty() {
            0
        } else {
            wrapper_bytes + item_bytes
        };

        PackedContext {
            items: packed,
            decisions,
            total_bytes,
            budget_bytes: config.total_budget_bytes,
        }
    }
}

fn fit_packed_item(
    item: &ContextItem,
    max_rendered_bytes: usize,
    max_preview_bytes: usize,
) -> Option<(String, ContextPackAction, usize)> {
    let content = item.content.as_deref().unwrap_or("");
    let max_content_bytes = content.len().min(max_preview_bytes.max(1));
    let can_be_ref_only = item.large_ref.is_some();

    let mut best: Option<(String, ContextPackAction, usize)> = None;
    let mut low = 0usize;
    let mut high = max_content_bytes;
    while low <= high {
        let mid = (low + high) / 2;
        let preview = truncate_utf8(content, mid).to_string();
        let action = if preview.len() < content.len() {
            ContextPackAction::Previewed
        } else {
            ContextPackAction::Selected
        };
        let candidate = packed_item_from_context(item, preview.clone(), 0);
        let rendered_bytes = render_packed_item(&candidate).len();
        if rendered_bytes <= max_rendered_bytes {
            best = Some((preview, action, rendered_bytes));
            low = mid.saturating_add(1);
        } else if mid == 0 {
            break;
        } else {
            high = mid - 1;
        }
    }

    best.and_then(|(preview, action, rendered_bytes)| {
        if preview.len() >= 64 || preview.len() == content.len() || can_be_ref_only {
            Some((preview, action, rendered_bytes))
        } else {
            None
        }
    })
}

fn packed_item_from_context(
    item: &ContextItem,
    content: String,
    estimated_bytes: usize,
) -> PackedContextItem {
    PackedContextItem {
        id: item.id.clone(),
        source: item.source.clone(),
        kind: item.kind.clone(),
        name: item.name.clone(),
        content,
        content_type: item.content_type.clone(),
        large_ref: item.large_ref.clone(),
        estimated_bytes,
    }
}

fn render_packed_item(item: &PackedContextItem) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "<context_item id=\"{}\" kind=\"{:?}\" source=\"{:?}\" name=\"{}\">\n",
        escape_attr(&item.id),
        item.kind,
        item.source,
        escape_attr(&item.name)
    ));
    if let Some(content_type) = &item.content_type {
        out.push_str(&format!(
            "<content_type>{}</content_type>\n",
            escape_text(content_type)
        ));
    }
    if let Some(large_ref) = &item.large_ref {
        out.push_str(&format!(
            "<full_payload_ref id=\"{}\" expand_tool=\"{}\" />\n",
            escape_attr(&large_ref.id),
            escape_attr(&large_ref.expand_tool)
        ));
    }
    out.push_str("<content>\n");
    out.push_str(&escape_text(&item.content));
    out.push_str("\n</content>\n</context_item>\n");
    out
}

fn should_redact(item: &ContextItem, config: &ContextPackingConfig) -> bool {
    matches!(item.sensitivity, ContextSensitivity::Secret) && !config.include_secret
        || matches!(item.sensitivity, ContextSensitivity::Private) && !config.include_private
}

fn decision(
    item: &ContextItem,
    action: ContextPackAction,
    reason: impl Into<String>,
    score: f32,
    estimated_bytes: usize,
) -> ContextPackDecision {
    ContextPackDecision {
        id: item.id.clone(),
        kind: item.kind.clone(),
        action,
        reason: reason.into(),
        score,
        estimated_bytes,
    }
}

fn score_item(item: &ContextItem, query_terms: &BTreeSet<String>) -> f32 {
    let item_terms = tokenize(&item.searchable_text());
    let overlap = query_terms.intersection(&item_terms).count() as f32;
    let read_penalty = (item.read_count as f32).min(5.0);
    item.priority as f32 * 2.0 + overlap * 25.0 - read_penalty
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|term| term.len() >= 2)
        .map(|term| term.to_ascii_lowercase())
        .collect()
}

fn truncate_utf8(input: &str, max_bytes: usize) -> &str {
    if input.len() <= max_bytes {
        return input;
    }
    let mut end = max_bytes;
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    &input[..end]
}

fn escape_attr(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_text(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Query against a host/local context index.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub struct ContextSearchQuery {
    pub text: String,
    pub limit: usize,
    #[serde(default)]
    pub kinds: Vec<ContextKind>,
    #[serde(default)]
    pub sources: Vec<ContextSource>,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

impl ContextSearchQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 10,
            kinds: Vec::new(),
            sources: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct ContextSearchResult {
    pub item: ContextItem,
    pub score: f32,
}

/// Local retrieval abstraction. Embedding/vector stores can implement this
/// trait; the default in-memory index provides deterministic lexical behavior.
pub trait LocalContextIndex {
    fn upsert(&mut self, item: ContextItem);
    fn delete(&mut self, id: &str) -> bool;
    fn search(&self, query: &ContextSearchQuery) -> Vec<ContextSearchResult>;
    fn get(&self, id: &str) -> Option<&ContextItem>;
}

#[derive(Debug, Clone, Default)]
pub struct InMemoryContextIndex {
    items: BTreeMap<String, ContextItem>,
}

impl InMemoryContextIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl LocalContextIndex for InMemoryContextIndex {
    fn upsert(&mut self, item: ContextItem) {
        self.items.insert(item.id.clone(), item);
    }

    fn delete(&mut self, id: &str) -> bool {
        self.items.remove(id).is_some()
    }

    fn search(&self, query: &ContextSearchQuery) -> Vec<ContextSearchResult> {
        let query_terms = tokenize(&query.text);
        let limit = query.limit.max(1);
        let mut results = self
            .items
            .values()
            .filter(|item| query.kinds.is_empty() || query.kinds.contains(&item.kind))
            .filter(|item| query.sources.is_empty() || query.sources.contains(&item.source))
            .filter(|item| {
                query
                    .metadata
                    .iter()
                    .all(|(key, value)| item.metadata.get(key) == Some(value))
            })
            .map(|item| ContextSearchResult {
                item: item.clone(),
                score: score_item(item, &query_terms),
            })
            .filter(|result| result.score > 0.0 || query_terms.is_empty())
            .collect::<Vec<_>>();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.item.id.cmp(&b.item.id))
        });
        results.truncate(limit);
        results
    }

    fn get(&self, id: &str) -> Option<&ContextItem> {
        self.items.get(id)
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

    #[test]
    fn packs_relevant_context_under_total_and_kind_budgets() {
        let items = vec![
            ContextItem::new("web-1", ContextSource::Search, "Rust guide", "Ownership")
                .with_kind(ContextKind::WebExcerpt)
                .with_priority(1)
                .with_content("Rust ownership and borrowing details", "text/plain"),
            ContextItem::new("web-2", ContextSource::Search, "Cooking", "Pasta")
                .with_kind(ContextKind::WebExcerpt)
                .with_priority(10)
                .with_content("Pasta water and sauce", "text/plain"),
            ContextItem::new(
                "mem-1",
                ContextSource::Memory,
                "Project decision",
                "Use Rust",
            )
            .with_kind(ContextKind::MemoryFact)
            .with_priority(5)
            .with_content("The project uses Rust for the agent harness", "text/plain"),
        ];
        let config = ContextPackingConfig {
            total_budget_bytes: 700,
            default_kind_budget_bytes: 500,
            default_kind_max_items: 4,
            max_item_preview_bytes: 250,
            kind_budgets: vec![
                ContextKindBudget::new(ContextKind::WebExcerpt, 300, 1),
                ContextKindBudget::new(ContextKind::MemoryFact, 300, 2),
            ],
            ..Default::default()
        };

        let packed = ContextPacker::pack(
            "How does the Rust harness handle ownership?",
            &items,
            &config,
        );

        assert!(packed.total_bytes <= config.total_budget_bytes);
        assert_eq!(packed.items.len(), 2);
        assert!(packed.items.iter().any(|item| item.id == "web-1"));
        assert!(packed.items.iter().any(|item| item.id == "mem-1"));
        assert!(packed.decisions.iter().any(
            |decision| decision.id == "web-2" && decision.action == ContextPackAction::Dropped
        ));
    }

    #[test]
    fn redacts_secret_context_by_default() {
        let items = vec![
            ContextItem::new("secret-1", ContextSource::File, "Env", "API key")
                .with_kind(ContextKind::FileExcerpt)
                .with_sensitivity(ContextSensitivity::Secret)
                .with_content("SECRET=abc", "text/plain"),
        ];

        let packed = ContextPacker::pack("api key", &items, &ContextPackingConfig::default());

        assert!(packed.items.is_empty());
        assert_eq!(packed.decisions[0].action, ContextPackAction::Redacted);
    }

    #[test]
    fn large_payload_refs_are_previewed_and_render_expand_handles() {
        let item = ContextItem::new(
            "tool-1",
            ContextSource::ToolResult,
            "Build log",
            "Cargo test",
        )
        .with_kind(ContextKind::ToolObservation)
        .with_large_ref(LargeContextRef::new("payload-1", "read_context_ref").with_bytes(50_000))
        .with_content(&"line ".repeat(500), "text/plain");
        let config = ContextPackingConfig {
            total_budget_bytes: 600,
            default_kind_budget_bytes: 600,
            max_item_preview_bytes: 200,
            ..Default::default()
        };

        let packed = ContextPacker::pack("cargo test failure", &[item], &config);
        let prompt = packed.to_prompt_block();

        assert_eq!(packed.items.len(), 1);
        assert_eq!(packed.decisions[0].action, ContextPackAction::Previewed);
        assert!(prompt.contains("full_payload_ref"));
        assert!(prompt.contains("read_context_ref"));
        assert!(packed.items[0].content.len() <= 200);
    }

    #[test]
    fn rendered_prompt_block_stays_within_budget_with_escaping_and_long_names() {
        let item = ContextItem::new(
            "xml-1",
            ContextSource::File,
            "Very long name with <xml> & \"quotes\" ".repeat(8),
            "XML-heavy content",
        )
        .with_kind(ContextKind::FileExcerpt)
        .with_large_ref(LargeContextRef::new("payload<&>1", "read_context_ref"))
        .with_content(&"<tag>&value</tag>".repeat(200), "text/xml");
        let config = ContextPackingConfig {
            total_budget_bytes: 900,
            default_kind_budget_bytes: 900,
            max_item_preview_bytes: 800,
            ..Default::default()
        };

        let packed = ContextPacker::pack("xml value", &[item], &config);
        let rendered = packed.to_prompt_block();

        assert!(rendered.len() <= config.total_budget_bytes);
        assert!(packed.total_bytes <= config.total_budget_bytes);
    }

    #[test]
    fn ref_only_items_can_be_packed_and_do_not_block_later_valid_duplicates() {
        let ref_only = ContextItem::new(
            "ref-only",
            ContextSource::ToolResult,
            "Full log",
            "No preview",
        )
        .with_kind(ContextKind::ToolObservation)
        .with_large_ref(LargeContextRef::new("payload-1", "read_context_ref"));
        let duplicate_with_preview = ContextItem::new(
            "ref-preview",
            ContextSource::ToolResult,
            "Full log preview",
            "Preview",
        )
        .with_kind(ContextKind::ToolObservation)
        .with_priority(5)
        .with_large_ref(LargeContextRef::new("payload-1", "read_context_ref"))
        .with_content("important failure preview", "text/plain");

        let packed = ContextPacker::pack(
            "failure preview",
            &[ref_only, duplicate_with_preview],
            &ContextPackingConfig::default(),
        );
        let prompt = packed.to_prompt_block();

        assert_eq!(packed.items.len(), 1);
        assert_eq!(packed.items[0].id, "ref-preview");
        assert!(prompt.contains("payload-1"));
    }

    #[test]
    fn cutready_like_web_reference_flood_packs_excerpt_and_externalizes_payload() {
        let huge_web_page = format!(
            "{}\n{}",
            "Relevant finding: Azure rejected request bodies near 64KB after web context.",
            "navigation boilerplate duplicated url content ".repeat(500)
        );
        let items = vec![
            ContextItem::new(
                "web-full",
                ContextSource::Search,
                "https://example.test/provider-byte-budget",
                "Fetched web page about provider byte budget",
            )
            .with_kind(ContextKind::WebExcerpt)
            .with_priority(10)
            .with_large_ref(
                LargeContextRef::new("web-payload-1", "read_context_ref")
                    .with_bytes(huge_web_page.len()),
            )
            .with_content(huge_web_page, "text/plain"),
            ContextItem::new(
                "web-duplicate-url",
                ContextSource::Search,
                "https://example.test/provider-byte-budget",
                "Duplicate URL-only search result",
            )
            .with_kind(ContextKind::WebExcerpt)
            .with_large_ref(LargeContextRef::new("web-payload-1", "read_context_ref"))
            .with_content("same payload should not be packed twice", "text/plain"),
            ContextItem::new(
                "memory-budget",
                ContextSource::Memory,
                "Provider byte budget decision",
                "Reserve provider overhead",
            )
            .with_kind(ContextKind::MemoryFact)
            .with_priority(4)
            .with_content(
                "Agentive reserves serialization and gateway overhead before sending Azure requests.",
                "text/plain",
            ),
        ];
        let config = ContextPackingConfig {
            total_budget_bytes: 1_200,
            default_kind_budget_bytes: 900,
            default_kind_max_items: 3,
            max_item_preview_bytes: 500,
            kind_budgets: vec![
                ContextKindBudget::new(ContextKind::WebExcerpt, 650, 2),
                ContextKindBudget::new(ContextKind::MemoryFact, 350, 2),
            ],
            ..Default::default()
        };

        let packed = ContextPacker::pack("Why did Azure reject the web context?", &items, &config);
        let prompt = packed.to_prompt_block();

        assert!(prompt.len() <= config.total_budget_bytes);
        assert!(prompt.contains("web-full"));
        assert!(prompt.contains("web-payload-1"));
        assert!(prompt.contains("read_context_ref"));
        assert!(prompt.contains("memory-budget"));
        assert!(prompt.contains("Relevant finding"));
        assert!(!prompt.contains(&"navigation boilerplate duplicated url content ".repeat(20)));
        assert!(packed.decisions.iter().any(|decision| {
            decision.id == "web-duplicate-url"
                && decision.action == ContextPackAction::Dropped
                && decision.reason == "duplicate full-payload reference"
        }));
    }

    #[test]
    fn mixed_memory_docs_and_chat_pack_deterministically_under_kind_budgets() {
        let items = vec![
            ContextItem::new(
                "doc-context",
                ContextSource::File,
                "Context architecture docs",
                "Typed context packing design",
            )
            .with_kind(ContextKind::ReferenceDoc)
            .with_priority(2)
            .with_content(
                "Typed context items are packed before provider serialization.",
                "text/markdown",
            ),
            ContextItem::new(
                "memory-agentive",
                ContextSource::Memory,
                "Project direction",
                "Make Agentive an excellent harness",
            )
            .with_kind(ContextKind::MemoryFact)
            .with_priority(5)
            .with_content(
                "Do not defer context orchestration work to Prompty.",
                "text/plain",
            ),
            ContextItem::new(
                "recent-turn",
                ContextSource::User,
                "Latest user request",
                "Run confidence work to ground",
            )
            .with_kind(ContextKind::RecentTurn)
            .with_priority(1)
            .with_content(
                "How can we drive confidence just with our own project?",
                "text/plain",
            ),
            ContextItem::new(
                "irrelevant-high-priority",
                ContextSource::Search,
                "Pasta",
                "Cooking",
            )
            .with_kind(ContextKind::WebExcerpt)
            .with_priority(30)
            .with_content("Pasta recipes and boiling water", "text/plain"),
        ];
        let config = ContextPackingConfig {
            total_budget_bytes: 1_500,
            default_kind_budget_bytes: 800,
            default_kind_max_items: 2,
            max_item_preview_bytes: 500,
            kind_budgets: vec![
                ContextKindBudget::new(ContextKind::MemoryFact, 300, 1),
                ContextKindBudget::new(ContextKind::ReferenceDoc, 400, 1),
                ContextKindBudget::new(ContextKind::RecentTurn, 400, 1),
                ContextKindBudget::new(ContextKind::WebExcerpt, 1, 0),
            ],
            ..Default::default()
        };

        let packed = ContextPacker::pack("Agentive context confidence harness", &items, &config);
        let ids = packed
            .items
            .iter()
            .map(|item| item.id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(ids, vec!["memory-agentive", "doc-context", "recent-turn"]);
        assert_eq!(
            packed.to_prompt_block(),
            "<context_pack>\n<context_item id=\"memory-agentive\" kind=\"MemoryFact\" source=\"Memory\" name=\"Project direction\">\n<content_type>text/plain</content_type>\n<content>\nDo not defer context orchestration work to Prompty.\n</content>\n</context_item>\n<context_item id=\"doc-context\" kind=\"ReferenceDoc\" source=\"File\" name=\"Context architecture docs\">\n<content_type>text/markdown</content_type>\n<content>\nTyped context items are packed before provider serialization.\n</content>\n</context_item>\n<context_item id=\"recent-turn\" kind=\"RecentTurn\" source=\"User\" name=\"Latest user request\">\n<content_type>text/plain</content_type>\n<content>\nHow can we drive confidence just with our own project?\n</content>\n</context_item>\n</context_pack>"
        );
        assert!(packed.decisions.iter().any(|decision| {
            decision.id == "irrelevant-high-priority"
                && decision.action == ContextPackAction::Dropped
        }));
    }

    #[test]
    fn in_memory_context_index_filters_and_ranks() {
        let mut index = InMemoryContextIndex::new();
        index.upsert(
            ContextItem::new("a", ContextSource::File, "Runner", "Context packer")
                .with_kind(ContextKind::FileExcerpt)
                .with_content("Runner integrates context packing", "text/rust")
                .with_metadata("project", "agentive"),
        );
        index.upsert(
            ContextItem::new("b", ContextSource::Search, "Pasta", "Cooking")
                .with_kind(ContextKind::WebExcerpt)
                .with_content("Pasta recipe", "text/plain")
                .with_metadata("project", "agentive"),
        );

        let mut query = ContextSearchQuery::new("runner context").with_limit(5);
        query.kinds.push(ContextKind::FileExcerpt);
        query.metadata.insert("project".into(), "agentive".into());
        let results = index.search(&query);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item.id, "a");
        assert!(results[0].score > 0.0);
        assert!(index.delete("a"));
        assert!(index.get("a").is_none());
    }
}
