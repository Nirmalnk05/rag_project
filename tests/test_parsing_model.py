from src.parsing.model import (
    FigureElement, ParsedDocument, TableElement, TextElement,
)


def test_element_types_are_fixed():
    assert TextElement("s", "t").type == "text"
    assert TableElement("s", "| a |").type == "table"
    assert FigureElement("s", "a diagram").type == "figure"


def test_type_field_is_not_a_constructor_arg():
    # `type` is derived, not passed in
    import pytest
    with pytest.raises(TypeError):
        TextElement("s", "t", type="table")  # type: ignore[call-arg]


def test_parsed_document_by_type():
    doc = ParsedDocument(
        title="Pump Manual", source_file="pump.pdf",
        elements=[
            TextElement("Intro", "hello", page=1),
            TableElement("Alarms", "| code | meaning |", page=2),
            FigureElement("Wiring", "a wiring diagram", image_ref="img1", page=3),
        ],
    )
    assert len(doc.by_type("text")) == 1
    assert len(doc.by_type("table")) == 1
    assert doc.by_type("figure")[0].caption == "a wiring diagram"
    assert doc.elements[0].page == 1
