"""Tests for Phase A embedding utilities."""

from train_embeddings import build_document_preview, infer_document_label


def test_infer_document_label_strips_versions_and_status_tokens() -> None:
    document = {
        "filename": "260205_Rapportage economische analyse zeef 1_v1.6_cs_lf_eo.docx",
        "id": "1",
    }

    assert infer_document_label(document) == "rapportage economische analyse zeef 1"


def test_infer_document_label_uses_id_when_filename_missing() -> None:
    document = {"id": "42", "text": "Voorbeeldtekst"}

    assert infer_document_label(document) == "document 42"


def test_build_document_preview_preserves_relevant_headings() -> None:
    long_text = (
        "Inhoudsopgave\n"
        "1. Inleiding\n"
        "2. Managementsamenvatting\n"
        "Managementsamenvatting\n"
        "Dit is de kernboodschap.\n"
        "Aanbevelingen\n"
        "Werk aanbeveling A uit.\n"
    )

    preview = build_document_preview(long_text, max_characters=120)

    assert "Managementsamenvatting" in preview
    assert "Aanbevelingen" in preview
    assert len(preview) <= 120
