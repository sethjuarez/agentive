//! Shared request-body byte limit handling for provider adapters.

use crate::error::AgentError;

/// Default Azure request body cap (64KB), safely below gateways that truncate
/// request bodies around ~79KB and then return JSON parse errors.
pub(super) const DEFAULT_AZURE_MAX_REQUEST_BYTES: usize = 64 * 1024;

pub(super) fn default_azure_max_request_bytes(endpoint: &str) -> usize {
    if endpoint.contains("azure.com") {
        DEFAULT_AZURE_MAX_REQUEST_BYTES
    } else {
        usize::MAX
    }
}

/// Build a provider request body, dropping oldest compactable items until the
/// serialized body fits within `max_request_bytes`.
///
/// Providers keep their own body shapes by supplying `make_body`, their own
/// prefix-preservation rule, and their own group-removal rule for tool-call
/// pairs. This keeps the byte-limit policy shared without flattening distinct
/// provider wire formats into one abstraction.
pub(super) fn compact_items_to_request_limit<T>(
    items: &mut Vec<T>,
    max_request_bytes: usize,
    provider_label: &str,
    item_label: &str,
    make_body: impl Fn(&[T]) -> Result<serde_json::Value, AgentError>,
    preserve_prefix_item: impl Fn(&T) -> bool,
    remove_group: impl Fn(&mut Vec<T>, usize) -> usize,
) -> Result<serde_json::Value, AgentError> {
    let body = make_body(items)?;
    let serialized = serde_json::to_string(&body)
        .map_err(|e| AgentError::Stream(format!("Failed to serialize request: {e}")))?;

    if serialized.len() <= max_request_bytes {
        return Ok(body);
    }

    let preserved_end = items
        .iter()
        .position(|item| !preserve_prefix_item(item))
        .unwrap_or(items.len());

    let mut dropped = 0usize;
    while items.len() > preserved_end + 2 {
        let trial = make_body(items)?;
        let size = serde_json::to_string(&trial)
            .map_err(|e| AgentError::Stream(format!("Failed to serialize request: {e}")))?
            .len();
        if size <= max_request_bytes {
            break;
        }

        let removed = remove_group(items, preserved_end);
        if removed == 0 {
            break;
        }
        dropped += removed;
    }

    let compacted = make_body(items)?;
    let compacted_size = serde_json::to_string(&compacted)
        .map_err(|e| AgentError::Stream(format!("Failed to serialize request: {e}")))?
        .len();

    if compacted_size > max_request_bytes {
        return Err(AgentError::Stream(format!(
            "{provider_label} request body is too large after compaction: {compacted_size} bytes exceeds {max_request_bytes} bytes"
        )));
    }

    if dropped > 0 {
        log::warn!(
            "[agentive] {provider_label} body exceeded {}KB — dropped {} {item_label}(s) to fit",
            max_request_bytes / 1024,
            dropped
        );
    }

    Ok(compacted)
}
