"""read_frame infers each column's type from the whole file."""

from alphapy.frame import read_frame


def test_a_numeric_column_empty_in_its_first_rows_is_read_as_numbers(tmp_path):
    # A long-lookback feature has no value until enough history exists; here the first 12,000 rows are empty.
    rows = ["x,late"] + [f"{i}," for i in range(12000)] + [f"{i},{i / 7:.4f}" for i in range(12000, 12010)]
    (tmp_path / "train.csv").write_text("\n".join(rows) + "\n")
    df = read_frame(str(tmp_path), "train", "csv", ",")
    assert df["late"].dtype.is_float()
    assert df["late"].null_count() == 12000 and abs(df["late"][-1] - 12009 / 7) < 1e-3
