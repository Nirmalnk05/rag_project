from openpyxl import Workbook

from src.parsing.base import rows_to_markdown
from src.parsing.excel_parser import ExcelParser


def _make_xlsx(path, sheets: dict[str, list[list]]):
    wb = Workbook()
    wb.remove(wb.active)
    for name, rows in sheets.items():
        ws = wb.create_sheet(title=name)
        for r in rows:
            ws.append(r)
    wb.save(path)


def test_rows_to_markdown_shape():
    md = rows_to_markdown(["code", "meaning"], [["4F", "low pressure"]])
    lines = md.splitlines()
    assert lines[0] == "| code | meaning |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| 4F | low pressure |"


def test_pipe_in_cell_escaped():
    md = rows_to_markdown(["a"], [["x|y"]])
    assert "x\\|y" in md


def test_sheet_becomes_table_element_with_section(tmp_path):
    p = tmp_path / "faults.xlsx"
    _make_xlsx(p, {"Faults": [["code", "meaning"], ["4F", "low suction"], ["E-24", "leak"]]})
    doc = ExcelParser().parse(p)
    assert doc.source_file == "faults.xlsx"
    tables = doc.by_type("table")
    assert len(tables) == 1
    assert tables[0].section == "Faults"
    assert "4F" in tables[0].markdown and "E-24" in tables[0].markdown


def test_large_sheet_splits_and_repeats_header(tmp_path):
    p = tmp_path / "big.xlsx"
    data = [["code", "meaning"]] + [[f"c{i}", f"m{i}"] for i in range(120)]
    _make_xlsx(p, {"S": data})
    doc = ExcelParser(rows_per_chunk=50).parse(p)
    tables = doc.by_type("table")
    assert len(tables) == 3  # 120 rows / 50
    for t in tables:
        assert t.markdown.splitlines()[0] == "| code | meaning |"  # header repeated


def test_blank_rows_dropped(tmp_path):
    p = tmp_path / "gappy.xlsx"
    _make_xlsx(p, {"S": [["h1", "h2"], [None, None], ["a", "b"]]})
    doc = ExcelParser().parse(p)
    md = doc.by_type("table")[0].markdown
    assert "| a | b |" in md
    assert "|  |  |" not in md  # blank row not emitted
