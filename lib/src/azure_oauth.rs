//! Azure Entra ID OAuth for keyless Azure OpenAI authentication.
//!
//! Provides the OAuth protocol logic for authenticating with Azure OpenAI
//! via Entra ID instead of API keys. Two flows are supported:
//!
//! - **Authorization Code + PKCE** (primary) — opens browser, redirects to localhost
//! - **Device Code** (fallback) — for headless environments
//!
//! This module is framework-agnostic. The calling application handles UX
//! (opening browser, showing device codes, etc.) and calls these functions.
//!
//! # Example (browser flow)
//! ```no_run
//! # async fn example() -> Result<(), String> {
//! use agentive::azure_oauth;
//!
//! // 1. Start the flow — get auth URL + localhost listener
//! let (init, verifier) = azure_oauth::start_auth_code_flow("organizations", None, None).await?;
//!
//! // 2. App opens init.auth_url in browser (app-specific UX)
//!
//! // 3. Wait for the redirect callback
//! let code = azure_oauth::wait_for_auth_code(init.port, 300, "My App").await?;
//!
//! // 4. Exchange code for tokens
//! let redirect_uri = format!("http://localhost:{}", init.port);
//! let tokens = azure_oauth::exchange_code_for_token(
//!     "organizations", &code, &redirect_uri, &verifier, None, None,
//! ).await?;
//!
//! // 5. Later, refresh the token
//! let fresh = azure_oauth::refresh_token(
//!     "organizations", tokens.refresh_token.as_deref().unwrap(), None, None,
//! ).await?;
//! # Ok(())
//! # }
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

/// Default scope for Azure OpenAI / AI Services.
pub const AZURE_OPENAI_SCOPE: &str = "https://ai.azure.com/.default offline_access";

/// Scope for Azure Resource Manager (ARM) API — subscription/resource discovery.
pub const AZURE_MANAGEMENT_SCOPE: &str =
    "https://management.azure.com/.default offline_access";

/// Azure PowerShell first-party client ID — works for cognitive services scopes.
pub const DEFAULT_CLIENT_ID: &str = "1950a258-227b-4e31-a9cf-717495945fc2";

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Token response from the Azure token endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    #[serde(default)]
    pub refresh_token: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Returned to the caller so it can open the browser with `auth_url`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCodeFlowInit {
    /// The full authorization URL to open in the browser.
    pub auth_url: String,
    /// The localhost port the callback server is listening on.
    pub port: u16,
}

/// Initial response from the device code endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCodeResponse {
    pub device_code: String,
    pub user_code: String,
    pub verification_uri: String,
    pub expires_in: u64,
    pub interval: u64,
    pub message: String,
}

// ---------------------------------------------------------------------------
// PKCE helpers
// ---------------------------------------------------------------------------

/// Generate a cryptographic random code_verifier (64 chars, unreserved URI chars).
fn generate_code_verifier() -> String {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut rng = rand::rng();
    (0..64)
        .map(|_| {
            let idx = rng.random_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Derive the S256 code_challenge from a code_verifier.
fn code_challenge(verifier: &str) -> String {
    use base64::Engine;
    let hash = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hash)
}

/// Extract a query parameter value from a URL path like `/?code=abc&state=xyz`.
fn extract_query_param(path: &str, key: &str) -> Option<String> {
    let query = path.split('?').nth(1)?;
    for pair in query.split('&') {
        let mut kv = pair.splitn(2, '=');
        if kv.next()? == key {
            let val = kv.next().unwrap_or("");
            return Some(urlencoding::decode(val).unwrap_or_default().into_owned());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Global listener storage
// ---------------------------------------------------------------------------

/// Pending TCP listeners keyed by port, kept alive between start and wait.
fn pending_listeners() -> &'static tokio::sync::Mutex<std::collections::HashMap<u16, TcpListener>>
{
    static INSTANCE: std::sync::OnceLock<
        tokio::sync::Mutex<std::collections::HashMap<u16, TcpListener>>,
    > = std::sync::OnceLock::new();
    INSTANCE.get_or_init(|| tokio::sync::Mutex::new(std::collections::HashMap::new()))
}

// ---------------------------------------------------------------------------
// Authorization Code + PKCE flow
// ---------------------------------------------------------------------------

/// Start the authorization code flow with PKCE.
///
/// Binds a localhost listener for the redirect, generates PKCE codes,
/// and returns the auth URL + code verifier. The caller should:
/// 1. Open `init.auth_url` in the browser
/// 2. Call [`wait_for_auth_code`] to receive the authorization code
/// 3. Call [`exchange_code_for_token`] to get tokens
pub async fn start_auth_code_flow(
    tenant_id: &str,
    client_id: Option<&str>,
    scope: Option<&str>,
) -> Result<(AuthCodeFlowInit, String), String> {
    let cid = client_id.unwrap_or(DEFAULT_CLIENT_ID);
    let scp = scope.unwrap_or(AZURE_OPENAI_SCOPE);

    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .map_err(|e| format!("Failed to bind localhost listener: {e}"))?;
    let port = listener
        .local_addr()
        .map_err(|e| format!("Failed to get local address: {e}"))?
        .port();

    let redirect_uri = format!("http://localhost:{port}");
    let verifier = generate_code_verifier();
    let challenge = code_challenge(&verifier);

    let auth_url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/authorize?\
         client_id={}&response_type=code&redirect_uri={}&scope={}&\
         code_challenge={}&code_challenge_method=S256&prompt=select_account",
        tenant_id,
        cid,
        urlencoding::encode(&redirect_uri),
        urlencoding::encode(scp),
        challenge,
    );

    // Store the listener so wait_for_auth_code can retrieve it
    {
        let mut map = pending_listeners().lock().await;
        map.insert(port, listener);
    }

    let init = AuthCodeFlowInit { auth_url, port };
    Ok((init, verifier))
}

/// Wait for the browser to redirect back to our localhost server.
///
/// Extracts the authorization code from the query string.
/// Shows a success/error page in the browser with the given `app_name`.
pub async fn wait_for_auth_code(
    port: u16,
    timeout_secs: u64,
    app_name: &str,
) -> Result<String, String> {
    let listener = {
        let mut map = pending_listeners().lock().await;
        map.remove(&port)
            .ok_or_else(|| format!("No pending listener on port {port}"))?
    };

    let timeout = std::time::Duration::from_secs(timeout_secs);
    let (mut stream, _addr) = tokio::time::timeout(timeout, listener.accept())
        .await
        .map_err(|_| "Timed out waiting for browser redirect".to_string())?
        .map_err(|e| format!("Accept failed: {e}"))?;

    // Read the HTTP request
    let mut buf = vec![0u8; 4096];
    let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
        .await
        .map_err(|e| format!("Failed to read request: {e}"))?;
    let request = String::from_utf8_lossy(&buf[..n]);

    let first_line = request.lines().next().unwrap_or("");
    let path = first_line.split_whitespace().nth(1).unwrap_or("");

    // Check for error response
    if path.contains("error=") {
        let error_desc = extract_query_param(path, "error_description")
            .unwrap_or_else(|| extract_query_param(path, "error").unwrap_or_default());
        let error_page = format!(
            "{}\r\n\r\n{}",
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close",
            error_html(app_name, &error_desc)
        );
        let _ = stream.write_all(error_page.as_bytes()).await;
        let _ = stream.shutdown().await;
        return Err(format!("Auth error: {error_desc}"));
    }

    let code = extract_query_param(path, "code")
        .ok_or_else(|| format!("No authorization code in redirect: {path}"))?;

    let success_page = format!(
        "{}\r\n\r\n{}",
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nConnection: close",
        success_html(app_name)
    );
    let _ = stream.write_all(success_page.as_bytes()).await;
    let _ = stream.shutdown().await;

    Ok(code)
}

/// Exchange an authorization code for tokens.
///
/// The `scope` parameter controls which resource the token is minted for.
/// Pass `None` to use the default Azure OpenAI scope.
pub async fn exchange_code_for_token(
    tenant_id: &str,
    code: &str,
    redirect_uri: &str,
    code_verifier: &str,
    client_id: Option<&str>,
    scope: Option<&str>,
) -> Result<TokenResponse, String> {
    let cid = client_id.unwrap_or(DEFAULT_CLIENT_ID);
    let scp = scope.unwrap_or(AZURE_OPENAI_SCOPE);
    let url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        tenant_id
    );

    let http = reqwest::Client::new();
    let resp = http
        .post(&url)
        .form(&[
            ("client_id", cid),
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("code_verifier", code_verifier),
            ("scope", scp),
        ])
        .send()
        .await
        .map_err(|e| format!("Token exchange failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token exchange error: {body}"));
    }

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse token response: {e}"))
}

// ---------------------------------------------------------------------------
// Refresh token
// ---------------------------------------------------------------------------

/// Refresh an access token using a refresh token.
///
/// Pass a different `scope` to swap the token audience (e.g. switch
/// from ARM management scope to AI Services scope using the same
/// refresh token).
pub async fn refresh_token(
    tenant_id: &str,
    refresh_token: &str,
    client_id: Option<&str>,
    scope: Option<&str>,
) -> Result<TokenResponse, String> {
    let cid = client_id.unwrap_or(DEFAULT_CLIENT_ID);
    let scp = scope.unwrap_or(AZURE_OPENAI_SCOPE);
    let url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        tenant_id
    );

    let http = reqwest::Client::new();
    let resp = http
        .post(&url)
        .form(&[
            ("client_id", cid),
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("scope", scp),
        ])
        .send()
        .await
        .map_err(|e| format!("Token refresh failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Token refresh error: {body}"));
    }

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse refresh response: {e}"))
}

// ---------------------------------------------------------------------------
// Device Code flow (fallback)
// ---------------------------------------------------------------------------

/// Error during token polling.
#[derive(Debug, Deserialize)]
struct TokenErrorResponse {
    error: String,
    #[allow(dead_code)]
    error_description: Option<String>,
}

/// Request a device code from Azure Entra ID.
pub async fn request_device_code(
    tenant_id: &str,
    client_id: Option<&str>,
    scope: Option<&str>,
) -> Result<DeviceCodeResponse, String> {
    let cid = client_id.unwrap_or(DEFAULT_CLIENT_ID);
    let scp = scope.unwrap_or(AZURE_OPENAI_SCOPE);
    let url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/devicecode",
        tenant_id
    );

    let http = reqwest::Client::new();
    let resp = http
        .post(&url)
        .form(&[("client_id", cid), ("scope", scp)])
        .send()
        .await
        .map_err(|e| format!("Device code request failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Device code endpoint error: {body}"));
    }

    resp.json()
        .await
        .map_err(|e| format!("Failed to parse device code response: {e}"))
}

/// Poll the token endpoint until the user completes device code authentication.
pub async fn poll_for_token(
    tenant_id: &str,
    device_code: &str,
    interval_secs: u64,
    timeout_secs: u64,
    client_id: Option<&str>,
) -> Result<TokenResponse, String> {
    let cid = client_id.unwrap_or(DEFAULT_CLIENT_ID);
    let url = format!(
        "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
        tenant_id
    );

    let http = reqwest::Client::new();
    let interval = std::time::Duration::from_secs(interval_secs.max(5));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

    loop {
        if std::time::Instant::now() > deadline {
            return Err("Device code flow timed out".into());
        }
        tokio::time::sleep(interval).await;

        let resp = http
            .post(&url)
            .form(&[
                ("client_id", cid),
                ("grant_type", "urn:ietf:params:oauth:grant-type:device_code"),
                ("device_code", device_code),
            ])
            .send()
            .await
            .map_err(|e| format!("Token poll failed: {e}"))?;

        let status = resp.status();
        let body = resp.text().await.map_err(|e| format!("Read error: {e}"))?;

        if status.is_success() {
            return serde_json::from_str(&body)
                .map_err(|e| format!("Failed to parse token: {e}"));
        }

        if let Ok(err) = serde_json::from_str::<TokenErrorResponse>(&body) {
            match err.error.as_str() {
                "authorization_pending" => continue,
                "slow_down" => {
                    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                    continue;
                }
                "expired_token" => return Err("Device code expired".into()),
                other => return Err(format!("Token error: {other}")),
            }
        }
        return Err(format!("Unexpected response ({status}): {body}"));
    }
}

// ---------------------------------------------------------------------------
// HTML templates
// ---------------------------------------------------------------------------

fn success_html(app_name: &str) -> String {
    format!(
        "<!DOCTYPE html>\
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<title>{app_name} — Signed In</title>\
<style>\
*{{margin:0;padding:0;box-sizing:border-box}}\
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;\
display:flex;align-items:center;justify-content:center;min-height:100vh;\
background:#f8f9fa;color:#1a1a1a}}\
@media(prefers-color-scheme:dark){{body{{background:#1a1a1a;color:#e8e8e8}}}}\
.card{{text-align:center;padding:48px 40px;max-width:420px;\
background:#fff;border-radius:16px;box-shadow:0 2px 24px rgba(0,0,0,.08)}}\
@media(prefers-color-scheme:dark){{.card{{background:#2a2a2a;box-shadow:0 2px 24px rgba(0,0,0,.3)}}}}\
.icon{{font-size:48px;margin-bottom:16px}}\
h1{{font-size:20px;font-weight:600;margin-bottom:8px}}\
p{{font-size:14px;opacity:.7;line-height:1.5}}\
.fade{{animation:fadeIn .4s ease}}\
@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}\
</style></head>\
<body><div class=\"card fade\">\
<div class=\"icon\">✅</div>\
<h1>Signed in to {app_name}</h1>\
<p>You can close this tab and return to the app.</p>\
</div>\
<script>setTimeout(()=>window.close(),3000)</script>\
</body></html>"
    )
}

fn error_html(app_name: &str, detail: &str) -> String {
    format!(
        "<!DOCTYPE html>\
<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\
<title>{app_name} — Sign-in Failed</title>\
<style>\
*{{margin:0;padding:0;box-sizing:border-box}}\
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;\
display:flex;align-items:center;justify-content:center;min-height:100vh;\
background:#f8f9fa;color:#1a1a1a}}\
@media(prefers-color-scheme:dark){{body{{background:#1a1a1a;color:#e8e8e8}}}}\
.card{{text-align:center;padding:48px 40px;max-width:420px;\
background:#fff;border-radius:16px;box-shadow:0 2px 24px rgba(0,0,0,.08)}}\
@media(prefers-color-scheme:dark){{.card{{background:#2a2a2a;box-shadow:0 2px 24px rgba(0,0,0,.3)}}}}\
.icon{{font-size:48px;margin-bottom:16px}}\
h1{{font-size:20px;font-weight:600;margin-bottom:8px}}\
p{{font-size:14px;opacity:.7;line-height:1.5}}\
.detail{{margin-top:12px;font-size:12px;opacity:.5;word-break:break-word}}\
.fade{{animation:fadeIn .4s ease}}\
@keyframes fadeIn{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:none}}}}\
</style></head>\
<body><div class=\"card fade\">\
<div class=\"icon\">❌</div>\
<h1>Sign-in failed</h1>\
<p>Something went wrong during authentication.</p>\
<p class=\"detail\">{detail}</p>\
</div></body></html>"
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_verifier_length_and_charset() {
        let v = generate_code_verifier();
        assert_eq!(v.len(), 64);
        for c in v.chars() {
            assert!(
                c.is_ascii_alphanumeric() || c == '-' || c == '.' || c == '_' || c == '~',
                "unexpected char: {c}"
            );
        }
    }

    #[test]
    fn test_code_verifier_uniqueness() {
        let a = generate_code_verifier();
        let b = generate_code_verifier();
        assert_ne!(a, b, "two verifiers should differ");
    }

    #[test]
    fn test_code_challenge_s256() {
        // Known test vector: verifier "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        // SHA256 → base64url = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let challenge = code_challenge(verifier);
        assert_eq!(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn test_extract_query_param() {
        assert_eq!(
            extract_query_param("/?code=abc123&state=xyz", "code"),
            Some("abc123".into())
        );
        assert_eq!(
            extract_query_param("/?code=abc123&state=xyz", "state"),
            Some("xyz".into())
        );
        assert_eq!(
            extract_query_param("/?code=abc123", "missing"),
            None
        );
        assert_eq!(
            extract_query_param("/no-query", "code"),
            None
        );
    }

    #[test]
    fn test_extract_query_param_url_encoded() {
        assert_eq!(
            extract_query_param("/?error_description=access%20denied", "error_description"),
            Some("access denied".into())
        );
    }

    #[test]
    fn test_success_html_contains_app_name() {
        let html = success_html("TestApp");
        assert!(html.contains("TestApp"));
        assert!(html.contains("Signed in to TestApp"));
    }

    #[test]
    fn test_error_html_contains_detail() {
        let html = error_html("TestApp", "bad credentials");
        assert!(html.contains("TestApp"));
        assert!(html.contains("bad credentials"));
    }

    #[tokio::test]
    async fn test_start_auth_code_flow_returns_url_and_port() {
        let (init, verifier) = start_auth_code_flow("organizations", None, None)
            .await
            .unwrap();

        assert!(init.auth_url.contains("login.microsoftonline.com"));
        assert!(init.auth_url.contains("organizations"));
        assert!(init.auth_url.contains(DEFAULT_CLIENT_ID));
        assert!(init.auth_url.contains("code_challenge_method=S256"));
        assert!(init.port > 0);
        assert_eq!(verifier.len(), 64);

        // Clean up the pending listener
        let mut map = pending_listeners().lock().await;
        map.remove(&init.port);
    }

    #[tokio::test]
    async fn test_start_auth_code_flow_custom_client() {
        let (init, _) = start_auth_code_flow("mytenant", Some("custom-client"), None)
            .await
            .unwrap();

        assert!(init.auth_url.contains("mytenant"));
        assert!(init.auth_url.contains("custom-client"));

        let mut map = pending_listeners().lock().await;
        map.remove(&init.port);
    }

    #[tokio::test]
    async fn test_wait_for_auth_code_no_pending_listener() {
        let result = wait_for_auth_code(9999, 1, "Test").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No pending listener"));
    }
}
