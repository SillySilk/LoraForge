from core.caption_replace import replace_text, replace_in_set, scan_matches


def test_literal_regex_chars():
    assert replace_text("a (medium) shot", "(medium)", "close", False, True) == ("a close shot", 1)
    assert replace_text("score_7+ safe", "score_7+", "x", True, False) == ("x safe", 1)


def test_whole_word_skips_inside_words():
    assert replace_text("a man and a woman", "man", "boy", False, True) == ("a boy and a woman", 1)
    assert replace_text("a man and a woman", "man", "boy", False, False) == ("a boy and a woboy", 2)


def test_match_case():
    assert replace_text("Red red RED", "red", "blue", True, True) == ("Red blue RED", 1)
    assert replace_text("Red red RED", "red", "blue", False, True) == ("blue blue blue", 3)


def test_blank_find_is_noop():
    assert replace_text("1girl, solo", "", "X", False, False) == ("1girl, solo", 0)
    assert replace_text("1girl, solo", "   ", "X", False, False) == ("1girl, solo", 0)


def test_delete_tidies_separators():
    assert replace_text("1girl, 1boy, solo", "1boy", "", False, True) == ("1girl, solo", 1)


def test_no_match_returns_original():
    assert replace_text("1girl,solo", "cat", "", False, True) == ("1girl,solo", 0)


def test_replace_in_set(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"")
    (tmp_path / "a.txt").write_text("a man, solo", encoding="utf-8")
    (tmp_path / "a.nl").write_text("a man stands", encoding="utf-8")
    other = tmp_path / "b.png"
    other.write_bytes(b"")
    (tmp_path / "b.txt").write_text("a woman", encoding="utf-8")
    paths = [str(img), str(other)]

    assert scan_matches(paths, "man", False, True) == (2, 1)
    assert replace_in_set(paths, "man", "boy", False, True) == (2, 2)
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a boy, solo"
    assert (tmp_path / "a.nl").read_text(encoding="utf-8") == "a boy stands"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "a woman"
    assert not (tmp_path / "a.tags").exists()
