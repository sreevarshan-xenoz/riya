import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from PIL import Image

from colab.sprites import gen_expressions, gen_outfits
from colab.sprites.gen_expressions import (
    face_box, face_mask, job_list, parse_face_box, parse_only, run_jobs, stable_seed,
)
from colab.sprites.common import (
    EXPRESSION_PROMPTS, OUTFITS, VARIANT_PROMPTS, existing_pngs, load_pipeline, save_image_atomic,
)
from colab.sprites.review_sheet import make_sheet


def protocol_expressions():
    return json.loads(Path("docs/protocol.schema.json").read_text(encoding="utf-8"))["expressions"]


def test_prompts_cover_protocol_expressions_exactly():
    assert sorted(EXPRESSION_PROMPTS) == sorted(protocol_expressions())
    assert set(VARIANT_PROMPTS) == {"closed", "open", "blink"} and len(OUTFITS) == 4


def test_make_sheet_draws_a_labelled_grid(tmp_path):
    paths = []
    for i in range(5):
        p = tmp_path / f"t{i}.png"
        Image.new("RGB", (80, 120), (40 * i, 90, 160)).save(p)
        paths.append(p)
    out = make_sheet(paths, [f"label {i}" for i in range(5)], tmp_path / "sheet.png", cols=2)
    assert out == tmp_path / "sheet.png"
    with Image.open(out) as sheet:
        assert sheet.width > 160 and sheet.height > 3 * 120  # 2 cols x 3 rows plus label strips


def test_save_image_atomic_writes_png_without_leftovers(tmp_path):
    out = save_image_atomic(Image.new("RGB", (8, 8), "red"), tmp_path / "a.png")
    assert out == tmp_path / "a.png" and Image.open(out).size == (8, 8)
    assert [p.name for p in tmp_path.iterdir()] == ["a.png"]


def test_save_image_atomic_failure_leaves_no_partial_file(tmp_path):
    class Dying:
        def save(self, path, format=None):
            Path(path).write_bytes(b"half a file")
            raise OSError("session dropped")

    with pytest.raises(OSError):
        save_image_atomic(Dying(), tmp_path / "a.png")
    assert list(tmp_path.iterdir()) == []


def test_existing_pngs_ignores_tmp_leftovers(tmp_path):
    for name in ("outfit_a_1.png", "outfit_b_1.tmp.png", "other.png"):
        Image.new("RGB", (4, 4)).save(tmp_path / name)
    assert existing_pngs(tmp_path, "outfit_*.png") == {tmp_path / "outfit_a_1.png"}


def test_gen_outfits_resumes_without_torch_when_everything_exists(tmp_path):
    for oid in OUTFITS:
        for seed in (11, 22):
            Image.new("RGB", (32, 48)).save(tmp_path / f"outfit_{oid}_{seed}.png")
    gen_outfits.main(["--refs", str(tmp_path / "no refs needed"), "--out", str(tmp_path)])
    assert (tmp_path / "outfits_sheet.png").exists()


def test_common_imports_without_torch_or_diffusers():
    code = (
        "import sys, colab.sprites.common as c, colab.sprites.gen_outfits;"
        "assert callable(c.load_pipeline) and 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


ROOT = Path(__file__).resolve().parents[1]
BOX = (256, 60, 576, 420)


def test_face_box_default_and_override():
    assert face_box(832, 1216, None) == (256, 60, 576, 420)
    assert face_box(832, 1216, (1, 2, 3, 4)) == (1, 2, 3, 4)


def test_job_list_skips_existing_and_keeps_order():
    jobs = job_list(["neutral", "happy"], ["closed", "open", "blink"], {"neutral_closed.png", "happy_open.png"})
    assert jobs == [("neutral", "open"), ("neutral", "blink"), ("happy", "closed"), ("happy", "blink")]


def test_stable_seed_is_the_same_in_fresh_processes_with_different_hash_seeds():
    here = stable_seed("pouty", 5)
    assert here == 9126 + 5 and stable_seed("sad", 5) != here
    code = "from colab.sprites.gen_expressions import stable_seed; print(stable_seed('pouty', 5))"
    for hash_seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        out = subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env=env, capture_output=True, text=True)
        assert int(out.stdout) == here


def test_face_mask_is_white_on_the_box_black_elsewhere_and_feathered():
    mask = face_mask((832, 1216), BOX)
    assert mask.mode == "L" and mask.size == (832, 1216)
    assert mask.getpixel((416, 240)) == 255 and mask.getpixel((20, 1000)) == 0 and mask.getpixel((416, 800)) == 0
    assert 0 < mask.getpixel((256, 240)) < 255  # the edge is soft


def test_face_mask_extends_down_for_hands_and_clamps_to_the_image():
    plain, hands = face_mask((832, 1216), BOX), face_mask((832, 1216), BOX, extend_down=280)
    assert plain.getpixel((416, 600)) == 0 and hands.getpixel((416, 600)) == 255  # 420 + 280 = 700
    assert hands.getpixel((416, 800)) == 0
    low = face_mask((200, 300), (50, 100, 150, 250), extend_down=280)  # the extension runs past the bottom edge
    assert low.size == (200, 300) and low.getpixel((100, 299)) == 255


def test_parse_only_validates_against_the_expression_prompts():
    assert parse_only("pouty, sad") == ["pouty", "sad"]
    with pytest.raises(argparse.ArgumentTypeError, match="pouting"):
        parse_only("pouty,pouting")


def test_parse_face_box_needs_four_ordered_ints():
    assert parse_face_box("1, 2,30,40") == (1, 2, 30, 40)
    for bad in ("1,2,3", "a,b,c,d", "10,2,5,40"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_face_box(bad)


def test_main_rejects_unknown_only_names_before_doing_any_work(tmp_path, capsys):
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "white_tee", "seed": 1}))
    argv = ["--choice", str(tmp_path / "choice.json"), "--refs", str(tmp_path), "--out", str(tmp_path / "o"), "--only", "nope"]
    with pytest.raises(SystemExit) as exc:
        gen_expressions.main(argv)
    assert exc.value.code == 2 and "nope" in capsys.readouterr().err and not (tmp_path / "o").exists()


def test_main_rejects_a_choice_with_an_unknown_outfit(tmp_path):
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "tuxedo", "seed": 1}))
    with pytest.raises(SystemExit, match="tuxedo"):
        gen_expressions.main(["--choice", str(tmp_path / "choice.json"), "--refs", str(tmp_path), "--out", str(tmp_path)])


def test_help_works_without_torch_or_diffusers():
    code = (
        "import sys\n"
        "from colab.sprites import gen_expressions\n"
        "try: gen_expressions.main(['--help'])\n"
        "except SystemExit as e: assert e.code == 0\n"
        "assert 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, capture_output=True)
    script = subprocess.run([sys.executable, "colab/sprites/gen_expressions.py", "--help"], check=True, cwd=ROOT,
                            capture_output=True, text=True)
    assert "--choice" in script.stdout and "--only" in script.stdout


def fake_inpaint(calls):
    def inpaint(image, mask, prompt, strength, seed):
        calls.append({"prompt": prompt, "strength": strength, "seed": seed, "src": image.getpixel((416, 240))})
        out = image.copy()
        out.paste((len(calls), 0, 0), (300, 100, 500, 300))  # stamp the call number inside the mask
        return out
    return inpaint


def grey_base(folder):
    Image.new("RGB", (832, 1216), (128, 128, 128)).save(folder / "base.png")


def test_run_jobs_chains_variants_from_closed_and_adds_a_hands_pass(tmp_path):
    grey_base(tmp_path)
    calls = []
    run_jobs([("happy", "closed"), ("happy", "open"), ("covering_eyes", "closed")], tmp_path, "white_tee", 7, BOX,
             fake_inpaint(calls))
    assert sorted(p.name for p in tmp_path.glob("*_*.png")) == ["covering_eyes_closed.png", "happy_closed.png", "happy_open.png"]
    assert [c["strength"] for c in calls] == [0.55, 0.55, 0.55, 0.75]
    assert calls[0]["src"] == (128, 128, 128)  # closed comes from base
    assert calls[1]["src"] == (1, 0, 0)  # open comes from happy_closed, not base
    assert calls[2]["src"] == (128, 128, 128) and calls[3]["src"] == (3, 0, 0)  # the hands pass runs on the face pass
    assert calls[0]["seed"] == calls[1]["seed"] == stable_seed("happy", 7)
    assert calls[2]["seed"] == calls[3]["seed"] == stable_seed("covering_eyes", 7)
    assert "closed mouth" in calls[0]["prompt"] and "open mouth" in calls[1]["prompt"]
    assert OUTFITS["white_tee"] in calls[0]["prompt"] and EXPRESSION_PROMPTS["happy"] in calls[0]["prompt"]


def test_run_jobs_generates_a_missing_closed_first_and_never_redoes_finished_frames(tmp_path):
    grey_base(tmp_path)
    calls = []
    run_jobs([("sad", "blink")], tmp_path, "white_tee", 1, BOX, fake_inpaint(calls))
    assert (tmp_path / "sad_closed.png").exists() and (tmp_path / "sad_blink.png").exists() and len(calls) == 2
    run_jobs([("sad", "closed"), ("sad", "blink")], tmp_path, "white_tee", 1, BOX, fake_inpaint(calls))
    assert len(calls) == 2


def test_run_jobs_keeps_everything_outside_the_mask_pixel_identical(tmp_path):
    grey_base(tmp_path)

    def repaint_everything(image, mask, prompt, strength, seed):
        return Image.new("RGB", image.size, (255, 0, 0))

    run_jobs([("happy", "closed")], tmp_path, "white_tee", 1, BOX, repaint_everything)
    with Image.open(tmp_path / "happy_closed.png") as frame:
        assert frame.getpixel((416, 240)) == (255, 0, 0) and frame.getpixel((100, 1000)) == (128, 128, 128)


def test_make_sheet_leaves_missing_frames_blank(tmp_path):
    Image.new("RGB", (80, 120), (200, 30, 30)).save(tmp_path / "a.png")
    out = make_sheet([tmp_path / "a.png", None], ["a", "missing"], tmp_path / "s.png", cols=2)
    with Image.open(out) as sheet:
        assert sheet.getpixel((8 + 128, 8 + 186)) == (200, 30, 30)  # centre of the first cell
        assert sheet.getpixel((8 + 256 + 8 + 128, 8 + 186)) == (32, 32, 36)  # the second cell stays background


def test_main_resumes_without_torch_and_writes_a_14_by_3_review_sheet(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "white_tee", "seed": 3}))
    Image.new("RGB", (64, 96)).save(out / "base.png")
    for expr in EXPRESSION_PROMPTS:
        for variant in VARIANT_PROMPTS:
            Image.new("RGB", (64, 96), (90, 90, 200)).save(out / f"{expr}_{variant}.png")
    gen_expressions.main(["--choice", str(tmp_path / "choice.json"), "--refs", str(tmp_path), "--out", str(out)])
    with Image.open(out / "review_sheet.png") as sheet:
        assert sheet.size == (3 * (256 + 8) + 8, 14 * (372 + 28 + 8) + 8)
