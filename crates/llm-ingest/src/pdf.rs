use std::path::Path;

use crate::error::IngestError;

/// Extract text from a PDF file.
pub fn extract_pdf(path: &Path) -> Result<String, IngestError> {
    pdf_extract::extract_text(path).map_err(|e| IngestError::Pdf {
        path: path.to_path_buf(),
        source: e.into(),
    })
}

/// Extract text from PDF bytes (for memory-mapped large files).
pub fn extract_pdf_bytes(data: &[u8], path: &Path) -> Result<String, IngestError> {
    pdf_extract::extract_text_from_mem(data).map_err(|e| IngestError::Pdf {
        path: path.to_path_buf(),
        source: e.into(),
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn extract_nonexistent_returns_error() {
        let result = super::extract_pdf(std::path::Path::new("/nonexistent.pdf"));
        assert!(result.is_err());
    }
}
