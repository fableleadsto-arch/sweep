"""Real local documents and data: permissions, preservation, limits and exports."""

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pytest

from sweep.desktop import data_tools as data


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "people.csv"
    path.write_text("name,age,city\nAsha,24,Kathmandu\nDev,17,Pokhara\nAsha,24,Kathmandu\nMira,,Kathmandu\n", encoding="utf-8")
    return path


def inspect(path, **kwargs):
    return data.inspect_data(str(path), [str(path)], **kwargs)


def process(path, request, tmp_path):
    return data.process_data(str(path), [str(path)], request, tmp_path / "artifacts")


def test_csv_profile_has_real_statistics_and_bounded_preview(csv_file):
    before = csv_file.read_bytes()
    result = inspect(csv_file)
    assert result["columns"] == ["name", "age", "city"]
    assert result["profile"]["rows"] == 4
    assert result["profile"]["duplicate_rows"] == 1
    assert result["profile"]["missing"]["age"] == 1
    assert result["profile"]["numeric_summary"]["age"] == {
        "count": 3, "min": 17.0, "max": 24.0, "mean": pytest.approx(65 / 3), "median": 24.0,
    }
    assert result["execution"] == "local"
    assert result["source_unchanged"] is True
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    assert csv_file.read_bytes() == before


def test_exact_file_grant_required_for_inspection_and_transformation(csv_file, tmp_path):
    for grants in ([], [str(tmp_path)], [str(tmp_path / "another.csv")]):
        with pytest.raises(PermissionError, match="Attach this file"):
            data.inspect_data(str(csv_file), grants)
        with pytest.raises(PermissionError):
            data.process_data(str(csv_file), grants, "convert to json", tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_file_byte_limit_is_enforced_before_parsing(csv_file, monkeypatch):
    monkeypatch.setattr(data, "MAX_BYTES", 5)
    with pytest.raises(ValueError, match="20 MB"):
        inspect(csv_file)


@pytest.mark.parametrize("header", ["a,a", ",b", "a," + "x" * 251])
def test_ambiguous_column_names_are_rejected(tmp_path, header):
    path = tmp_path / "bad.csv"
    path.write_text(header + "\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique, non-empty"):
        inspect(path)


def test_ragged_csv_is_not_silently_truncated(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Row 2 has 3 cells; expected 2"):
        inspect(path)


def test_unterminated_csv_quoting_is_not_silently_repaired(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text('name,note\nAsha,"unfinished\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        inspect(path)


def test_utf16_tsv_with_quoted_multiline_cell(tmp_path):
    path = tmp_path / "notes.tsv"
    path.write_text('name\tnote\nMira\t"two\nlines"\n', encoding="utf-16", newline="")
    result = inspect(path)
    assert result["rows"] == [["Mira", "two\nlines"]]


def test_blank_lines_do_not_consume_row_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_ROWS", 2)
    path = tmp_path / "blanks.csv"
    path.write_text("a\n\n1\n\n2\n\n", encoding="utf-8")
    assert inspect(path)["profile"]["rows"] == 2
    path.write_text("a\n1\n2\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="20,000 rows"):
        inspect(path)


@pytest.mark.parametrize("suffix,content", [
    (".json", '[{"a":1},{"b":2,"a":3}]'),
    (".json", '{"rows":[{"a":1},{"b":2,"a":3}]}'),
    (".jsonl", '{"a":1}\n\n{"b":2,"a":3}\n'),
    (".ndjson", '{"a":1}\n{"b":2,"a":3}\n'),
])
def test_json_variants_union_columns_and_missing_cells(tmp_path, suffix, content):
    path = tmp_path / ("data" + suffix)
    path.write_text(content, encoding="utf-8")
    result = inspect(path)
    assert result["columns"] == ["a", "b"]
    assert result["rows"] == [[1, None], [3, 2]]


@pytest.mark.parametrize("content", ['[1,2]', '{"rows":false}', '[{"a":NaN}]', '[{"a":Infinity}]', '{'])
def test_invalid_and_nonfinite_json_is_rejected(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        inspect(path)


def test_nested_json_values_are_serialized_without_execution(tmp_path):
    path = tmp_path / "nested.json"
    path.write_text(json.dumps([{"value": {"command": "delete everything"}}]), encoding="utf-8")
    assert inspect(path)["rows"] == [['{"command":"delete everything"}']]


@pytest.mark.parametrize("output_format", ["JSON", "JSONL"])
def test_structured_json_cells_retain_types_in_export(tmp_path, output_format):
    records = [{"details": {"tags": ["a", "b"], "enabled": True}, "items": [1, None, 3]}]
    path = tmp_path / "structured.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    result = process(path, "convert to " + output_format, tmp_path)
    output = Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8")
    loaded = json.loads(output) if output_format == "JSON" else [json.loads(line) for line in output.splitlines()]
    assert loaded == records


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_duplicate_json_keys_are_not_silently_discarded(tmp_path, suffix):
    path = tmp_path / ("duplicate" + suffix)
    path.write_text('{"name":"Asha","name":"Dev"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate object key"):
        process(path, "convert to csv", tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_profile_stays_finite_for_large_valid_numbers(tmp_path):
    path = tmp_path / "large.json"
    path.write_text('[{"x":1e308},{"x":1e308}]', encoding="utf-8")
    statistics = inspect(path)["profile"]["numeric_summary"]["x"]
    assert statistics["mean"] == 1e308
    assert statistics["median"] == 1e308
    json.dumps(statistics, allow_nan=False)


@pytest.mark.parametrize("phrase,operation", [
    ("summarize this data", "inspect"),
    ("What is in this file?", "inspect"),
    ("", "inspect"),
    ("Please remove duplicates", "deduplicate"),
    ("Remove duplicate rows", "deduplicate"),
    ("drop duplicates by name, city", "deduplicate"),
    ("clean this dataset", "clean"),
    ("filter where age >= 18", "filter"),
    ("filter age >= 18", "filter"),
    ("keep rows where city = Kathmandu", "filter"),
    ("convert this CSV to JSON", "convert"),
    ("convert this to a JSON file", "convert"),
    ("export as TSV", "convert"),
])
def test_common_requests_map_to_finite_operations(phrase, operation):
    assert data.parse_data_request(phrase)["operation"] == operation


@pytest.mark.parametrize("phrase", [
    "delete rows", "run SQL SELECT * FROM users", "sort by age", "normalize the age column",
    "deduplicate and delete the original", "clean and sort by age", "remove duplicates and filter where age > 18",
    "filter where age >= 18 and age < 30", "filter where age >= 18; DROP TABLE users",
    "convert to exe", "export as JSON and delete original", "convert filter where age >= 18 to json",
    "clean this dataset and fill missing ages with 0", "clean and replace Kathmandu with Pokhara",
])
def test_unsupported_and_ambiguous_operations_fail_clearly(phrase):
    with pytest.raises(ValueError):
        data.parse_data_request(phrase)


def test_quoted_column_and_literal_value_are_supported():
    result = data.parse_data_request('filter where "full name" = "Asha and Dev"')
    assert result["column"] == "full name"
    assert result["value"] == "Asha and Dev"


def test_filter_numeric_comparison_exports_new_file_and_preserves_source(csv_file, tmp_path):
    original = csv_file.read_bytes()
    result = process(csv_file, "filter where age >= 18 and export to JSON", tmp_path)
    assert (result["input_rows"], result["output_rows"]) == (4, 2)
    artifact = result["artifacts"][0]
    output = Path(artifact["path"])
    assert output.parent == tmp_path / "artifacts"
    assert output != csv_file
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"name": "Asha", "age": "24", "city": "Kathmandu"},
        {"name": "Asha", "age": "24", "city": "Kathmandu"},
    ]
    assert artifact["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert artifact["bytes"] == output.stat().st_size
    assert csv_file.read_bytes() == original


@pytest.mark.parametrize("condition,expected", [
    ("city contains MANDU", 3), ("CITY is Kathmandu", 3), ("age != 24", 2), ("age < 18", 1), ("age = 24.0", 2),
    ("age is at least 18", 2), ("age is less than 18", 1), ("age greater than or equal to 24", 2),
    ("city is not Kathmandu", 1), ("city equals Kathmandu", 3), ("age does not equal 24", 2),
])
def test_text_and_numeric_filters(csv_file, tmp_path, condition, expected):
    assert process(csv_file, "filter where " + condition, tmp_path)["output_rows"] == expected


def test_unknown_column_and_invalid_numeric_filter_do_not_create_artifact(csv_file, tmp_path):
    with pytest.raises(ValueError, match="was not found"):
        process(csv_file, "filter where missing = 1", tmp_path)
    with pytest.raises(ValueError, match="numeric comparison"):
        process(csv_file, "filter where age > old", tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_remove_duplicates_uses_selected_columns_and_keeps_first(csv_file, tmp_path):
    result = process(csv_file, "remove duplicates by city and export to jsonl", tmp_path)
    rows = [json.loads(line) for line in Path(result["artifacts"][0]["path"]).read_text(encoding="utf-8").splitlines()]
    assert [row["name"] for row in rows] == ["Asha", "Dev"]
    assert result["operation_details"]["columns"] == ["city"]


def test_clean_trims_strings_removes_empty_and_duplicate_rows(tmp_path):
    path = tmp_path / "messy.csv"
    path.write_text("name,value\n Asha , 1 \nAsha,1\n , \nDev,2\n", encoding="utf-8")
    result = process(path, "clean this file", tmp_path)
    assert result["rows"] == [["Asha", "1"], ["Dev", "2"]]
    assert (result["input_rows"], result["output_rows"]) == (4, 2)


@pytest.mark.parametrize("output_format", ["CSV", "TSV", "JSON", "JSONL"])
def test_conversion_round_trip_preserves_cell_content(tmp_path, output_format):
    path = tmp_path / "quotes.csv"
    path.write_text('name,note\nAsha,"comma, quote "" and\nnew line"\n', encoding="utf-8")
    result = process(path, "convert to " + output_format, tmp_path)
    assert inspect(Path(result["artifacts"][0]["path"]))["rows"] == inspect(path)["rows"]


def test_exports_use_unique_names(csv_file, tmp_path):
    first = process(csv_file, "convert to json", tmp_path)["artifacts"][0]["path"]
    second = process(csv_file, "convert to json", tmp_path)["artifacts"][0]["path"]
    assert first != second
    assert Path(first).exists() and Path(second).exists()


def test_summary_creates_no_artifacts(csv_file, tmp_path):
    result = process(csv_file, "summarize this data", tmp_path)
    assert result["operation"] == "inspect"
    assert "artifacts" not in result
    assert not (tmp_path / "artifacts").exists()


def database(path, statements):
    with sqlite3.connect(path) as connection:
        for statement in statements:
            connection.execute(statement)
    return path


def test_sqlite_single_table_inspection_and_export_never_change_database(tmp_path):
    path = database(tmp_path / "people.sqlite", ["CREATE TABLE people(name TEXT, age INTEGER)", "INSERT INTO people VALUES('Asha', 24), ('Dev', 17)"])
    before = path.read_bytes()
    result = inspect(path)
    assert result["table"] == "people"
    assert result["rows"] == [["Asha", 24], ["Dev", 17]]
    filtered = process(path, "filter where age >= 18", tmp_path)
    assert filtered["output_rows"] == 1
    assert path.read_bytes() == before
    assert not path.with_name(path.name + "-journal").exists()


def test_sqlite_multiple_tables_require_explicit_choice(tmp_path):
    path = database(tmp_path / "multiple.db", ["CREATE TABLE a(id INT)", "CREATE TABLE b(name TEXT)", "INSERT INTO b VALUES('Mira')"])
    result = inspect(path)
    assert result["needs_table"] is True
    assert [table["name"] for table in result["tables"]] == ["a", "b"]
    assert inspect(path, table="b")["rows"] == [["Mira"]]
    assert process(path, "inspect table 'b'", tmp_path)["rows"] == [["Mira"]]
    assert not (tmp_path / "artifacts").exists()


def test_sqlite_quoted_identifier_is_escaped_not_executed(tmp_path):
    odd_name = 'x"; DROP TABLE safe; --'
    identifier = '"' + odd_name.replace('"', '""') + '"'
    path = database(tmp_path / "names.db", [f"CREATE TABLE {identifier}(value TEXT)", f"INSERT INTO {identifier} VALUES('kept')", "CREATE TABLE safe(id INT)"])
    assert inspect(path, table=odd_name)["rows"] == [["kept"]]
    assert len(inspect(path)["tables"]) == 2
    with pytest.raises(ValueError, match="Choose an ordinary table"):
        inspect(path, table="safe; DROP TABLE safe")


def test_sqlite_views_are_never_selected(tmp_path):
    path = database(tmp_path / "view.db", ["CREATE VIEW v AS SELECT randomblob(1000000000)"])
    result = inspect(path)
    assert result["no_tables"] is True
    assert "no ordinary tables" in result["message"]
    with pytest.raises(ValueError, match="ordinary table"):
        inspect(path, table="v")


def test_sqlite_generated_expressions_are_not_evaluated(tmp_path):
    path = database(tmp_path / "generated.db", [
        "CREATE TABLE expressions(x INTEGER, result TEXT AS (printf('%1000000000s', x)) VIRTUAL)",
    ])
    with pytest.raises(ValueError, match="generated columns"):
        inspect(path)


def test_sqlite_commented_virtual_tables_and_shadow_tables_are_not_selected(tmp_path):
    path = database(tmp_path / "virtual.db", [
        "CREATE /* comment */ VIRTUAL TABLE documents USING fts5(content)",
    ])
    result = inspect(path)
    assert result["no_tables"] is True
    assert result["tables"] == [{"name": "documents", "available": False}]


def test_sqlite_binary_previews_cannot_be_mistaken_for_exported_values(tmp_path):
    path = database(tmp_path / "binary.db", [
        "CREATE TABLE attachments(content BLOB)", "INSERT INTO attachments VALUES(x'0001FF')",
    ])
    assert inspect(path)["rows"] == [["[binary: 3 bytes]"]]
    with pytest.raises(ValueError, match="binary cells"):
        process(path, "convert to json", tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_sqlite_wal_mode_is_rejected_as_incomplete_snapshot(tmp_path):
    path = database(tmp_path / "live.db", ["CREATE TABLE test(x INT)"])
    raw = bytearray(path.read_bytes())
    raw[18:20] = b"\x02\x02"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="standalone SQLite backup"):
        inspect(path)


def test_sqlite_row_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_ROWS", 1)
    path = database(tmp_path / "rows.db", ["CREATE TABLE test(x INT)", "INSERT INTO test VALUES(1),(2)"])
    with pytest.raises(ValueError, match="20,000 rows"):
        inspect(path)


def document(path, xml):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", xml)
    return path


def test_docx_extracts_paragraphs_and_table_cells_locally(tmp_path):
    path = document(tmp_path / "document.docx", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello </w:t></w:r><w:r><w:t>Sweep</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Cell</w:t></w:r></w:p></w:tc></w:tr></w:tbl></w:body></w:document>')
    result = data.inspect_document(str(path), [str(path)])
    assert result["text"] == "Hello Sweep\nCell"
    assert result["paragraphs_extracted"] == 2
    assert result["truncated"] is False


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_docx_xml_entities_are_rejected_even_with_utf16_encoding(tmp_path, encoding):
    xml = '<?xml version="1.0" encoding="' + encoding + '"?><!DOCTYPE document [<!ENTITY danger "expanded">]><document>&danger;</document>'
    path = document(tmp_path / "entities.docx", xml.encode(encoding))
    with pytest.raises(ValueError, match="unsupported entities"):
        data.inspect_document(str(path), [str(path)])


def test_text_preview_is_explicitly_truncated_and_requires_grant(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_TEXT", 12)
    path = tmp_path / "notes.md"
    path.write_text("A long local text with additional content", encoding="utf-8")
    with pytest.raises(PermissionError):
        data.inspect_document(str(path), [])
    result = data.inspect_document(str(path), [str(path)])
    assert result["text"] == "A long local"
    assert result["truncated"] is True
    assert "not the complete document" in result["message"]


def make_pdf(path, text="Sweep local PDF test"):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 700 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def test_pdf_extracts_real_selectable_text_and_page_provenance(tmp_path):
    path = tmp_path / "document.pdf"
    make_pdf(path)
    result = data.inspect_document(str(path), [str(path)])
    assert result["page_count"] == 1
    assert result["pages"] == [{"page": 1, "text": "Sweep local PDF test"}]
    assert "Page 1\nSweep local PDF test" == result["text"]


def test_pdf_truncation_is_reported_even_on_last_page(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "MAX_TEXT", 8)
    path = tmp_path / "long.pdf"
    make_pdf(path)
    result = data.inspect_document(str(path), [str(path)])
    assert result["truncated"] is True
    assert result["pages"][0]["text"] == "Sweep lo"


def test_pdf_compressed_stream_is_bounded_before_full_expansion(tmp_path, monkeypatch):
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    stream = DecodedStreamObject()
    stream.set_data(b" " * 5000)
    page[NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    path = tmp_path / "compressed.pdf"
    writer.write(path)
    assert path.stat().st_size < 2000
    monkeypatch.setattr(data, "MAX_BYTES", 2000)
    with pytest.raises(ValueError, match="exceeds local processing limits"):
        data.inspect_document(str(path), [str(path)])


def test_pdf_encryption_does_not_prompt_for_or_guess_passwords(tmp_path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt("test-only-password")
    path = tmp_path / "locked.pdf"
    writer.write(path)
    with pytest.raises(ValueError, match="Unlock this PDF locally"):
        data.inspect_document(str(path), [str(path)])


def test_network_share_input_and_artifacts_are_rejected():
    with pytest.raises(ValueError, match="network share"):
        data.inspect_data(r"\\server\share\data.csv", [r"\\server\share\data.csv"])
    with pytest.raises(ValueError, match="local task storage"):
        data._artifact(r"\\server\share\exports", ["a"], [{"a": 1}], "json")


def test_artifact_symlink_is_rejected(csv_file, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    artifacts = tmp_path / "artifacts"
    try:
        artifacts.symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this machine")
    with pytest.raises(ValueError, match="must not redirect"):
        process(csv_file, "convert to json", tmp_path)
    assert list(elsewhere.iterdir()) == []
