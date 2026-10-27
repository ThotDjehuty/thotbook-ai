//! M6 arXiv reference-closure crawler.
//!
//! Breadth-first crawl of the arXiv citation graph:
//!
//! 1. Metadata + abstract come from the official export API
//!    (`http://export.arxiv.org/api/query?id_list=...`, Atom XML, batched to
//!    100 ids/call, rate-limited to arXiv's 3-second etiquette).
//! 2. Full text comes from the ar5iv HTML mirror
//!    (`https://ar5iv.labs.arxiv.org/html/<id>`, with a fallback to
//!    `https://arxiv.org/html/<id>`), rendered to readable text with LaTeX
//!    equations preserved from the `<annotation encoding="application/x-tex">`
//!    elements.
//! 3. New-style + old-style arXiv ids referenced by each paper become the next
//!    BFS frontier, bounded by `max_depth` and `max_papers`.

use std::collections::{BTreeSet, HashSet, VecDeque};
use std::time::Duration;

use quick_xml::events::Event;
use quick_xml::Reader;
use reqwest::Client;
use scraper::{Html, Selector};
use tracing::{info, warn};

use crate::error::CorpusError;
use crate::nlab::extract_arxiv_ids;

/// Parsed arXiv metadata (Atom `<entry>`).
#[derive(Debug, Clone, Default)]
pub struct ArxivMeta {
    pub arxiv_id: String,
    pub title: String,
    pub authors: String,
    pub categories: String,
    pub primary_category: String,
    pub published_date: String,
    pub updated_date: String,
    pub abstract: String,
    pub pdf_url: String,
}

/// A fully crawled paper (metadata + optional full text + citation edges).
#[derive(Debug, Clone)]
pub struct ArxivDoc {
    pub arxiv_id: String,
    pub title: String,
    pub authors: String,
    pub categories: String,
    pub primary_category: String,
    pub published_date: String,
    pub abstract: String,
    pub full_text: String,
    pub cited_ids: Vec<String>,
    pub fetch_status: String,
    pub indexed_at: String,
    pub depth: usize,
}

/// Crawl configuration (bounded by design — "all of arXiv" is not a bounded
/// input; the closure from a seed set is).
#[derive(Debug, Clone)]
pub struct CrawlConfig {
    pub seeds: Vec<String>,
    /// Hard cap on papers ingested (0 = unlimited).
    pub max_papers: usize,
    /// BFS depth: 0 = seeds only, 1 = seeds + their references, ...
    pub max_depth: usize,
    /// Fetch full text only for papers at depth <= this (metadata beyond).
    pub fulltext_depth: usize,
    /// Concurrent ar5iv fetches.
    pub concurrency: usize,
    pub indexed_at: String,
}

impl Default for CrawlConfig {
    fn default() -> Self {
        Self {
            seeds: Vec::new(),
            max_papers: 5_000,
            max_depth: 1,
            fulltext_depth: 1,
            concurrency: 8,
            indexed_at: "2026-08-10".into(),
        }
    }
}

/// Normalize an arXiv id (strip URL prefixes and trailing version suffixes).
pub fn normalize_id(raw: &str) -> String {
    let mut s = raw.trim().to_string();
    for prefix in [
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "https://ar5iv.labs.arxiv.org/html/",
    ] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest.to_string();
        }
    }
    // strip version suffix for new-style ids (1601.05956v3 -> 1601.05956)
    let re = fancy_regex::RegexBuilder::new(r"^(\d{4}\.\d{4,5})(v\d+)?$")
        .build()
        .expect("static version regex");
    if let Ok(Some(caps)) = re.captures(&s) {
        if let Some(base) = caps.get(1) {
            return base.as_str().to_string();
        }
    }
    s
}

// ─────────────────────────── arXiv API metadata ───────────────────────────

fn decode_attr(value: &[u8]) -> String {
    String::from_utf8_lossy(value).into_owned()
}

/// Parse an Atom feed returned by `export.arxiv.org/api/query`.
fn parse_feed(xml: &str) -> Vec<ArxivMeta> {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut entries: Vec<ArxivMeta> = Vec::new();
    let mut cur = ArxivMeta::default();
    let mut buf = String::new();
    let mut in_author = false;
    let mut in_entry = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(e)) => {
                match e.local_name().as_ref() {
                    b"entry" => {
                        in_entry = true;
                        cur = ArxivMeta::default();
                    }
                    b"author" => in_author = true,
                    _ => {}
                }
                if in_entry && e.local_name().as_ref() == b"category" {
                    let mut term = String::new();
                    for a in e.attributes().flatten() {
                        if a.key.as_ref() == b"term" {
                            term = decode_attr(&a.value);
                        }
                    }
                    if !cur.categories.is_empty() {
                        cur.categories.push(',');
                    }
                    cur.categories.push_str(&term);
                }
                if in_entry && e.local_name().as_ref() == b"link" {
                    let mut title = String::new();
                    let mut href = String::new();
                    for a in e.attributes().flatten() {
                        match a.key.as_ref() {
                            b"title" => title = decode_attr(&a.value),
                            b"href" => href = decode_attr(&a.value),
                            _ => {}
                        }
                    }
                    if title == "pdf" {
                        cur.pdf_url = href;
                    }
                }
            }
            Ok(Event::Empty(e)) => {
                if in_entry && e.local_name().as_ref() == b"link" {
                    let mut title = String::new();
                    let mut href = String::new();
                    for a in e.attributes().flatten() {
                        match a.key.as_ref() {
                            b"title" => title = decode_attr(&a.value),
                            b"href" => href = decode_attr(&a.value),
                            _ => {}
                        }
                    }
                    if title == "pdf" {
                        cur.pdf_url = href;
                    }
                }
            }
            Ok(Event::Text(t)) | Ok(Event::CData(t)) => {
                if in_entry {
                    buf.push_str(&t.unescape().unwrap_or_default());
                }
            }
            Ok(Event::End(e)) => {
                match e.local_name().as_ref() {
                    b"id" => {
                        let v = buf.trim();
                        for prefix in ["http://arxiv.org/abs/", "http://arxiv.org/abs"] {
                            if let Some(rest) = v.strip_prefix(prefix) {
                                cur.arxiv_id = rest.to_string();
                                break;
                            }
                        }
                        if cur.arxiv_id.is_empty() {
                            cur.arxiv_id = v.to_string();
                        }
                    }
                    b"title" => cur.title = buf.trim().to_string(),
                    b"summary" => cur.abstract = buf.trim().to_string(),
                    b"name" if in_author => {
                        if !cur.authors.is_empty() {
                            cur.authors.push_str(", ");
                        }
                        cur.authors.push_str(buf.trim());
                    }
                    b"published" => cur.published_date = buf.trim().to_string(),
                    b"updated" => cur.updated_date = buf.trim().to_string(),
                    b"primary_category" => cur.primary_category = buf.trim().to_string(),
                    b"author" => in_author = false,
                    b"entry" => {
                        in_entry = false;
                        cur.arxiv_id = normalize_id(&cur.arxiv_id);
                        entries.push(std::mem::take(&mut cur));
                    }
                    _ => {}
                }
                buf.clear();
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }
    entries
}

/// Fetch metadata for a batch of ids (<= 100) from the arXiv export API.
async fn fetch_metadata(client: &Client, ids: &[String]) -> Result<Vec<ArxivMeta>, CorpusError> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let url = format!(
        "https://export.arxiv.org/api/query?id_list={}",
        ids.join(",")
    );
    info!(n = ids.len(), url = %url, "arxiv api query");
    // arXiv API etiquette: max 1 request / 3 s.
    tokio::time::sleep(Duration::from_secs(3)).await;
    let resp = client
        .get(&url)
        .send()
        .await
        .map_err(|source| CorpusError::Http { url: url.clone(), source })?;
    if !resp.status().is_success() {
        return Err(CorpusError::HttpStatus {
            url,
            status: resp.status().as_u16(),
        });
    }
    let body = resp
        .text()
        .await
        .map_err(|source| CorpusError::Http { url, source })?;
    Ok(parse_feed(&body))
}

// ─────────────────────────── ar5iv full text ───────────────────────────

fn tex_annotation_re() -> fancy_regex::Regex {
    fancy_regex::RegexBuilder::new(r"(?is)<annotation[^>]*>(.*?)</annotation>")
        .build()
        .expect("static annotation regex")
}

fn tag_re() -> fancy_regex::Regex {
    fancy_regex::Regex::new(r"(?s)<[^>]+>").expect("static tag regex")
}

fn block_re() -> fancy_regex::Regex {
    fancy_regex::RegexBuilder::new(r"(?i)</(?:p|li|div|section|h1|h2|h3|h4|table|tr|article)>")
        .build()
        .expect("static block regex")
}

fn script_re() -> fancy_regex::Regex {
    fancy_regex::RegexBuilder::new(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>")
        .build()
        .expect("static script/style regex")
}

fn unescape_html(s: &str) -> String {
    let mut out = s
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    // numeric character references
    let num_re = fancy_regex::Regex::new(r"&#(\d{1,6});|&#x([0-9a-fA-F]{1,6});")
        .expect("static numeric entity regex");
    out = num_re
        .replace_all(&out, |caps: &fancy_regex::Captures| {
            let code = caps
                .get(1)
                .and_then(|m| m.as_str().parse::<u32>().ok())
                .or_else(|| {
                    caps.get(2)
                        .and_then(|m| u32::from_str_radix(m.as_str(), 16).ok())
                });
            match code.and_then(char::from_u32) {
                Some(c) => c.to_string(),
                None => String::new(),
            }
        })
        .into_owned();
    out
}

/// Render an ar5iv HTML document to readable text. Paragraph prose is joined
/// with blank lines; TeX math annotations are kept inline as `$…$`.
fn html_to_text(html: &str) -> String {
    let doc = Html::parse_document(html);
    let para_sel = Selector::parse("p.ltx_p").unwrap_or_else(|_| Selector::parse("p").unwrap());
    let mut parts: Vec<String> = Vec::new();

    for p in doc.select(&para_sel) {
        let inner = p.inner_html();
        let mut s = script_re().replace_all(&inner, " ").into_owned();
        // keep the LaTeX source of math inline
        s = tex_annotation_re()
            .replace_all(&s, |caps: &fancy_regex::Captures| {
                let tex = caps
                    .get(1)
                    .map(|m| m.as_str().trim())
                    .unwrap_or_default();
                if tex.is_empty() {
                    String::new()
                } else {
                    format!(" $ {tex} $ ")
                }
            })
            .into_owned();
        s = block_re().replace_all(&s, "\n").into_owned();
        s = tag_re().replace_all(&s, " ").into_owned();
        let clean = unescape_html(&s);
        let words: Vec<&str> = clean.split_whitespace().collect();
        if !words.is_empty() {
            parts.push(words.join(" "));
        }
    }

    parts.join("\n\n")
}

/// Fetch and extract the full text for a single paper (ar5iv, arXiv HTML
/// fallback). Returns `(Some(text), "ar5iv"|"arxiv_html")` on success, or
/// `(None, status)` when no HTML source is available.
async fn fetch_fulltext(
    client: &Client,
    id: &str,
) -> Result<(Option<String>, &'static str), CorpusError> {
    let sources: [(&str, &str); 2] = [
        (
            &format!("https://ar5iv.labs.arxiv.org/html/{id}"),
            "ar5iv",
        ),
        (&format!("https://arxiv.org/html/{id}"), "arxiv_html"),
    ];
    for (url, label) in sources {
        let resp = match client.get(*url).send().await {
            Ok(r) => r,
            Err(e) => {
                warn!(id, url, error = %e, "fulltext fetch error");
                continue;
            }
        };
        if !resp.status().is_success() {
            continue;
        }
        let body = match resp.text().await {
            Ok(b) => b,
            Err(e) => {
                warn!(id, url, error = %e, "fulltext read error");
                continue;
            }
        };
        let text = html_to_text(&body);
        if text.split_whitespace().count() < 50 {
            continue; // too little content → not a usable rendering
        }
        return Ok((Some(text), label));
    }
    Ok((None, "no_html_source"))
}

// ─────────────────────────── BFS crawl ───────────────────────────

/// Breadth-first crawl of the arXiv reference closure from `seeds`.
pub async fn crawl(client: &Client, cfg: &CrawlConfig) -> Result<Vec<ArxivDoc>, CorpusError> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut frontier: VecDeque<(String, usize)> = VecDeque::new();
    for s in &cfg.seeds {
        let n = normalize_id(s);
        if !n.is_empty() && visited.insert(n.clone()) {
            frontier.push_back((n, 0));
        }
    }

    let mut docs: Vec<ArxivDoc> = Vec::new();
    let mut batch_no = 0usize;

    'outer: while !frontier.is_empty() {
        if cfg.max_papers > 0 && docs.len() >= cfg.max_papers {
            break;
        }
        batch_no += 1;

        let mut batch: Vec<(String, usize)> = Vec::new();
        while let Some(item) = frontier.pop_front() {
            batch.push(item);
            if batch.len() >= 100 {
                break;
            }
        }

        // Collect ids that are truly unvisited at this point.
        let pending: Vec<String> = batch
            .iter()
            .map(|(id, _)| id.clone())
            .filter(|id| visited.contains(id))
            .collect();
        let ids: Vec<String> = batch.iter().map(|(id, _)| id.clone()).collect();

        let metas = match fetch_metadata(client, &pending).await {
            Ok(m) => m,
            Err(e) => {
                warn!(batch = batch_no, error = %e, "metadata fetch failed; skipping batch");
                // still record pending as visited (done above) to avoid loops
                for id in &ids {
                    docs.push(ArxivDoc {
                        arxiv_id: id.clone(),
                        title: String::new(),
                        authors: String::new(),
                        categories: String::new(),
                        primary_category: String::new(),
                        published_date: String::new(),
                        abstract: String::new(),
                        full_text: String::new(),
                        cited_ids: Vec::new(),
                        fetch_status: "metadata_error".into(),
                        indexed_at: cfg.indexed_at.clone(),
                        depth: 0,
                    });
                }
                continue;
            }
        };

        let mut out: Vec<ArxivDoc> = Vec::with_capacity(metas.len());
        let mut tasks = tokio::task::JoinSet::new();

        for meta in metas {
            let depth = batch
                .iter()
                .find(|(id, _)| id == &meta.arxiv_id)
                .map(|(_, d)| *d)
                .unwrap_or(0);
            let fetch_full = depth <= cfg.fulltext_depth;
            let client = client.clone();
            let id = meta.arxiv_id.clone();
            tasks.spawn(async move {
                if !fetch_full {
                    return (
                        id,
                        meta,
                        (None, "metadata_only"),
                        Vec::<String>::new(),
                    );
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
                let (text, status) = fetch_fulltext(&client, &id).await.unwrap_or((None, "fetch_error"));
                let cited = text
                    .as_deref()
                    .map(extract_arxiv_ids)
                    .unwrap_or_default();
                (id, meta, (text, status), cited)
            });
        }

        let mut remaining = tasks.len();
        while remaining > 0 {
            // Cap in-flight concurrency.
            while tasks.len() > cfg.concurrency.min(1) {
                if let Some(res) = tasks.join_next().await {
                    remaining -= 1;
                    if let Ok((id, meta, (text, status), cited)) = res {
                        out.push(ArxivDoc {
                            arxiv_id: id,
                            title: meta.title,
                            authors: meta.authors,
                            categories: meta.categories,
                            primary_category: meta.primary_category,
                            published_date: meta.published_date,
                            abstract: meta.abstract,
                            full_text: text.unwrap_or_default(),
                            cited_ids: cited.clone(),
                            fetch_status: status.to_string(),
                            indexed_at: cfg.indexed_at.clone(),
                            depth: batch_depth(&batch, &meta.arxiv_id),
                        });
                        // enqueue references for the next BFS level
                        if batch_depth(&batch, &meta.arxiv_id) < cfg.max_depth {
                            for c in cited {
                                let n = normalize_id(&c);
                                if !n.is_empty() && visited.insert(n.clone()) {
                                    frontier.push_back((n, batch_depth(&batch, &meta.arxiv_id) + 1));
                                }
                            }
                        }
                        if cfg.max_papers > 0 && docs.len() + out.len() >= cfg.max_papers {
                            break;
                        }
                    }
                }
            }
            // finish the rest
            if let Some(res) = tasks.join_next().await {
                remaining -= 1;
                if let Ok((id, meta, (text, status), cited)) = res {
                    out.push(ArxivDoc {
                        arxiv_id: id,
                        title: meta.title,
                        authors: meta.authors,
                        categories: meta.categories,
                        primary_category: meta.primary_category,
                        published_date: meta.published_date,
                        abstract: meta.abstract,
                        full_text: text.unwrap_or_default(),
                        cited_ids: cited.clone(),
                        fetch_status: status.to_string(),
                        indexed_at: cfg.indexed_at.clone(),
                        depth: batch_depth(&batch, &meta.arxiv_id),
                    });
                    if batch_depth(&batch, &meta.arxiv_id) < cfg.max_depth {
                        for c in cited {
                            let n = normalize_id(&c);
                            if !n.is_empty() && visited.insert(n.clone()) {
                                frontier.push_back((n, batch_depth(&batch, &meta.arxiv_id) + 1));
                            }
                        }
                    }
                    if cfg.max_papers > 0 && docs.len() + out.len() >= cfg.max_papers {
                        break 'outer;
                    }
                }
            }
        }

        info!(batch = batch_no, written = out.len(), total = docs.len() + out.len(), "batch done");
        docs.extend(out);
    }

    if cfg.max_papers > 0 && docs.len() > cfg.max_papers {
        docs.truncate(cfg.max_papers);
    }
    Ok(docs)
}

fn batch_depth(batch: &[(String, usize)], id: &str) -> usize {
    batch
        .iter()
        .find(|(i, _)| i == id)
        .map(|(_, d)| *d)
        .unwrap_or(0)
}

/// Unique arXiv ids referenced by a set of crawled docs (for `--seeds` reuse).
pub fn union_cited(docs: &[ArxivDoc]) -> Vec<String> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for d in docs {
        for c in &d.cited_ids {
            set.insert(normalize_id(c));
        }
    }
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_urls_and_versions() {
        assert_eq!(normalize_id("https://arxiv.org/abs/1601.05956v3"), "1601.05956");
        assert_eq!(normalize_id("1601.05956"), "1601.05956");
        assert_eq!(normalize_id("hep-th/0512197"), "hep-th/0512197");
        assert_eq!(normalize_id("https://ar5iv.labs.arxiv.org/html/2301.12345"), "2301.12345");
    }

    #[test]
    fn parses_atom_feed() {
        let xml = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/1601.05956v3</id>
    <updated>2016-07-22T10:07:01Z</updated>
    <published>2016-01-22T13:15:02Z</published>
    <title>Supersymmetric supermanifolds and their deformations</title>
    <summary>We study the moduli of supermanifolds.</summary>
    <author><name>Albert Doe</name></author>
    <category term="math-ph" scheme="http://arxiv.org/schemas/atom"/>
    <category term="hep-th" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="math-ph" scheme="http://arxiv.org/schemas/atom"/>
    <link title="pdf" href="http://arxiv.org/pdf/1601.05956v3"/>
  </entry>
</feed>"#;
        let metas = parse_feed(xml);
        assert_eq!(metas.len(), 1);
        let m = &metas[0];
        assert_eq!(m.arxiv_id, "1601.05956");
        assert!(m.title.contains("supermanifolds"));
        assert!(m.authors.contains("Albert Doe"));
        assert!(m.abstract.contains("moduli"));
        assert!(m.categories.contains("math-ph") && m.categories.contains("hep-th"));
        assert_eq!(m.pdf_url, "http://arxiv.org/pdf/1601.05956v3");
    }

    #[test]
    fn renders_html_with_tex_math() {
        let html = r#"<html><body><article class="ltx_document">
<p class="ltx_p">Let <math><annotation encoding="application/x-tex">\omega</annotation></math>
be a symplectic form on a supermanifold.</p>
<p class="ltx_p">Then &amp; nothing else.</p>
</article></body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains(r"\omega"), "TeX math preserved: {text}");
        assert!(text.contains("symplectic form"));
        assert!(text.contains("nothing else"));
    }
}
