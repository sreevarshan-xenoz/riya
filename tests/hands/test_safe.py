"""Safe evaluators: calc, format_number, valid_url, safe_filename, unique_path."""
import pytest

from riya.core.hands.safe import calc, format_number, safe_filename, unique_path, valid_url

BS = chr(92)  # a backslash


@pytest.mark.parametrize("expr,want", [("12 * 7", 84), ("2^10", 1024), ("7 ÷ 2", 3.5), ("3 x 4", 12), ("3×4", 12),
                                       ("-(2+3)*2", -10), ("7 // 2", 3), ("7 % 4", 3), ("2 ** 0.5", 2 ** 0.5),
                                       ("  1.5 + 1 ", 2.5), ("10**99", 10**99)])
def test_calc_valid(expr, want):
    assert calc(expr) == want


@pytest.mark.parametrize("expr", [
    "__import__('os')", "(1).real", "a+1", "'x'*3", "2**101", "(10**99)*(10**99)", "1/0", "1" * 201, "",
    "True + 1", "1j", "[1]", "1 if 1 else 2", "abs(-1)", "(-8)**0.5", "1e999", "10**100 * 10", "2**-101", "~1",
    "0**-1", "1 << 3", "lambda: 1", "1; 2", "9" * 150, "1 +", "((((", None, 12, "5 // 0", "5 % 0", "\x00"])
def test_calc_rejects(expr):
    assert calc(expr) is None


def test_calc_bounds_each_step():
    assert calc("10**100") == 10**100 and calc("2**100") == 2**100
    assert calc("(10**60)*(10**60)/(10**60)") is None  # an intermediate is out of bounds even if the end isn't


def test_format_number():
    assert format_number(84.0) == "84" and format_number(3.5) == "3.5" and format_number(84) == "84"
    assert format_number(0.1 + 0.2) == "0.3" and format_number(-2.0) == "-2"


@pytest.mark.parametrize("text", [
    "file:///C:/x", "javascript:alert(1)", BS * 2 + "server" + BS + "share", "//server/share", "https://u:p@evil.com", " https://a.com",
    "https://a.com/x y", "https://a.com" + BS + "x","https://" + "a" * 2000 + ".com", "", "mailto:a@b.c", "ftp://a.com",
    "data:text/html,hi", "https://", "https://a.com:99999", "localhost", "https://evil.com\n", 'https://a.com/"x', None,
    "https://[::1]/", "https://exa_mple.com", "https://é.com", "C:" + BS + "Windows" + BS + "calc.exe", "calc.exe:/x"])
def test_valid_url_rejects(text):
    assert valid_url(text) is None


@pytest.mark.parametrize("text,want", [
    ("evil.com", "https://evil.com"), ("https://a.com/x?y=1#z", "https://a.com/x?y=1#z"),
    ("http://a.com", "http://a.com"), ("HTTPS://A.com", "HTTPS://A.com"), ("a.com:8080/x", "https://a.com:8080/x"),
    ("https://localhost:3000", "https://localhost:3000"), ("127.0.0.1", "https://127.0.0.1")])
def test_valid_url_accepts(text, want):
    assert valid_url(text) == want


@pytest.mark.parametrize("name,want", [
    ("../../etc/passwd", "etcpasswd.txt"), ("CON", "_CON.txt"), ("con.md", "_con.md"), ("notes.md", "notes.md"),
    ("a:b*c?.exe", "abc.exe.txt"), ("NUL.exe", "_NUL.exe.txt"), ("my list", "my list.txt"), ("Notes.TXT", "Notes.TXT"),
    ("x.log.", "x.log"), ("  spaced  ", "spaced.txt"), ("a" + BS + "b" + BS + "c.csv", "abc.csv"), ("COM1", "_COM1.txt")])
def test_safe_filename(name, want):
    assert safe_filename(name) == want


@pytest.mark.parametrize("name", ["...", "", "   ", "..", "///", None, 5])
def test_safe_filename_none(name):
    assert safe_filename(name) is None


def test_safe_filename_length():
    for n in ("a" * 500, "a" * 500 + ".md", "a" * 79 + ".exe"):
        out = safe_filename(n)
        assert out and len(out) <= 80 and out.endswith((".txt", ".md"))
    assert safe_filename("a" * 500 + ".md").endswith(".md")


def test_unique_path(tmp_path):
    assert unique_path(tmp_path, "a.txt") == tmp_path / "a.txt"
    (tmp_path / "a.txt").write_text("x")
    assert unique_path(tmp_path, "a.txt") == tmp_path / "a (2).txt"
    (tmp_path / "a (2).txt").write_text("x")
    assert unique_path(tmp_path, "a.txt") == tmp_path / "a (3).txt"
    assert unique_path(tmp_path, "../x.txt") is None


def test_unique_path_reapplies_the_name_rules(tmp_path):
    for bad in ("CON.txt", "a.txt" + chr(58) + "evil", "x/y.txt", "..", "", "a.exe", "NUL"):
        assert unique_path(tmp_path, bad) is None, bad
    assert unique_path(tmp_path, "_CON.txt") == tmp_path / "_CON.txt"
