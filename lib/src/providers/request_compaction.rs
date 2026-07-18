//! Shared request-body byte limit handling for provider adapters.

use crate::error::AgentError;

/// Default Azure request body cap (64KB), safely below gateways that truncate
/// request bodies around ~79KB and then return JSON parse errors.
pub(super) const DEFAULT_AZURE_MAX_REQUEST_BYTES: usize = 64 * 1024;
pub(super) const DEFAULT_AZURE_REQUEST_RESERVED_BYTES: usize = 2 * 1024;

pub(super) fn default_azure_max_request_bytes(endpoint: &str) -> usize {
    if endpoint.contains("azure.com") {
        DEFAULT_AZURE_MAX_REQUEST_BYTES
    } else {
        usize::MAX
    }
}

pub(super) fn default_azure_request_reserved_bytes(endpoint: &str) -> usize {
    if endpoint.contains("azure.com") {
        DEFAULT_AZURE_REQUEST_RESERVED_BYTES
    } else {
        0
    }
}

pub(super) fn effective_request_budget_bytes(
    max_request_bytes: usize,
    reserved_request_bytes: usize,
) -> usize {
    if max_request_bytes == usize::MAX || reserved_request_bytes == 0 {
        return max_request_bytes;
    }

    let reserved = reserved_request_bytes
        .min(max_request_bytes / 20)
        .min(max_request_bytes.saturating_sub(1));
    max_request_bytes.saturating_sub(reserved)
}

fn serialized_request_len(body: &serde_json::Value) -> Result<usize, AgentError> {
    serde_json::to_string(body)
        .map(|serialized| serialized.len())
        .map_err(|e| AgentError::Stream(format!("Failed to serialize request: {e}")))
}

fn oversized_request_error(
    provider_label: &str,
    compacted_size: usize,
    max_request_bytes: usize,
    effective_max_request_bytes: usize,
    reserved_request_bytes: usize,
) -> AgentError {
    let limit_detail = if effective_max_request_bytes < max_request_bytes {
        format!(
            "safe limit {effective_max_request_bytes} bytes (provider cap {max_request_bytes} bytes, {reserved_request_bytes} bytes reserved for serialization/gateway overhead)"
        )
    } else {
        format!("provider cap {max_request_bytes} bytes")
    };

    AgentError::Stream(format!(
        "{provider_label} request body is too large after compaction: {compacted_size} bytes exceeds {limit_detail}. Reduce attached files, references, web content, or earlier conversation context and try again."
    ))
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
    reserved_request_bytes: usize,
    provider_label: &str,
    item_label: &str,
    make_body: impl Fn(&[T]) -> Result<serde_json::Value, AgentError>,
    preserve_prefix_item: impl Fn(&T) -> bool,
    remove_group: impl Fn(&mut Vec<T>, usize) -> usize,
) -> Result<serde_json::Value, AgentError> {
    let effective_max_request_bytes =
        effective_request_budget_bytes(max_request_bytes, reserved_request_bytes);
    let body = make_body(items)?;
    let serialized_len = serialized_request_len(&body)?;

    if serialized_len <= effective_max_request_bytes {
        return Ok(body);
    }

    let preserved_end = items
        .iter()
        .position(|item| !preserve_prefix_item(item))
        .unwrap_or(items.len());

    let mut dropped = 0usize;
    while items.len() > preserved_end + 2 {
        let trial = make_body(items)?;
        let size = serialized_request_len(&trial)?;
        if size <= effective_max_request_bytes {
            break;
        }

        let removed = remove_group(items, preserved_end);
        if removed == 0 {
            break;
        }
        dropped += removed;
    }

    let compacted = make_body(items)?;
    let compacted_size = serialized_request_len(&compacted)?;

    if compacted_size > effective_max_request_bytes {
        return Err(oversized_request_error(
            provider_label,
            compacted_size,
            max_request_bytes,
            effective_max_request_bytes,
            max_request_bytes.saturating_sub(effective_max_request_bytes),
        ));
    }

    if dropped > 0 {
        log::warn!(
            "[agentive] {provider_label} body exceeded safe request budget ({}KB effective, {}KB cap) - dropped {} {item_label}(s) to fit",
            effective_max_request_bytes / 1024,
            max_request_bytes / 1024,
            dropped
        );
    }

    Ok(compacted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effective_budget_reserves_azure_overhead_without_collapsing_small_limits() {
        assert_eq!(
            effective_request_budget_bytes(
                DEFAULT_AZURE_MAX_REQUEST_BYTES,
                DEFAULT_AZURE_REQUEST_RESERVED_BYTES
            ),
            DEFAULT_AZURE_MAX_REQUEST_BYTES - DEFAULT_AZURE_REQUEST_RESERVED_BYTES
        );
        assert_eq!(
            effective_request_budget_bytes(1_000, DEFAULT_AZURE_REQUEST_RESERVED_BYTES),
            950
        );
        assert_eq!(
            effective_request_budget_bytes(usize::MAX, DEFAULT_AZURE_REQUEST_RESERVED_BYTES),
            usize::MAX
        );
        assert_eq!(effective_request_budget_bytes(1_000, 0), 1_000);
    }
}
