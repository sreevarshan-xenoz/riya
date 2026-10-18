import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft7Validator

from PIL import Image, ImageChops

from colab.sprites import build_pack as build_pack_module, gen_expressions, gen_outfits
from colab.sprites.build_pack import build_pack, normalise
from colab.sprites.gen_expressions import (
    DEFAULT_FACE_BOX, box_mask, frame_passes, frame_seed, hands_box, job_list, load_choice, merge_rerolls,
    parse_expressions, parse_face_box, read_run, run_jobs, stable_seed,
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


def test_load_refs_returns_top_anchored_square_crops(tmp_path):
    portrait = Image.new("RGB", (100, 250), (0, 0, 255))
    portrait.paste((255, 0, 0), (0, 0, 100, 100))  # the head is at the top
    portrait.save(tmp_path / "a portrait.png")
    landscape = Image.new("RGB", (300, 100), (0, 0, 255))
    landscape.paste((0, 255, 0), (100, 0, 200, 100))  # the centre square
    landscape.save(tmp_path / "b landscape.jpg", quality=95)
    Image.new("RGB", (64, 64), (9, 9, 9)).save(tmp_path / "c square.webp")
    (tmp_path / "notes.txt").write_text("not an image")
    refs = gen_outfits.load_refs(tmp_path)
    assert [r.size for r in refs] == [(100, 100), (100, 100), (64, 64)] and all(r.mode == "RGB" for r in refs)
    assert refs[0].getcolors() == [(100 * 100, (255, 0, 0))]
    green = refs[1].getpixel((50, 50))
    assert green[1] > 200 and green[0] < 50 and green[2] < 50 and refs[1].getpixel((2, 50))[2] < 100


def test_load_refs_without_images_exits(tmp_path):
    with pytest.raises(SystemExit, match="no reference images"):
        gen_outfits.load_refs(tmp_path)


def test_common_imports_without_torch_or_diffusers():
    code = (
        "import sys, colab.sprites.common as c, colab.sprites.gen_outfits;"
        "assert callable(c.load_pipeline) and 'torch' not in sys.modules and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


ROOT = Path(__file__).resolve().parents[1]
BOX = (256, 60, 576, 420)


def test_default_face_box():
    assert DEFAULT_FACE_BOX == (256, 60, 576, 420)


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


def test_box_mask_is_white_on_the_box_black_elsewhere_and_feathered():
    mask = box_mask((832, 1216), BOX)
    assert mask.mode == "L" and mask.size == (832, 1216)
    assert mask.getpixel((416, 240)) == 255 and mask.getpixel((20, 1000)) == 0 and mask.getpixel((416, 800)) == 0
    assert 0 < mask.getpixel((256, 240)) < 255  # the edge is soft
    low = box_mask((200, 300), (50, 100, 150, 400))  # a box running past the bottom edge is clamped
    assert low.size == (200, 300) and low.getpixel((100, 299)) == 255


def test_hands_box_is_1_8_face_widths_centred_extended_down_and_clamped():
    assert hands_box(BOX, (832, 1216)) == (128, 60, 704, 700)  # 576 wide around x=416, 420 + 280 below
    assert hands_box((600, 100, 800, 1100), (832, 1216)) == (520, 100, 832, 1216)
    assert hands_box((0, 0, 100, 100), (832, 1216))[0] == 0


def test_frame_passes_per_variant():
    face, hands = (BOX, 0.55), ((128, 60, 704, 700), 0.75)
    mouth, eyes = ((336, 276, 496, 420), 0.55), ((288, 168, 544, 258), 0.55)  # lower 40%, middle 50% / 30-55%, 80%
    assert frame_passes("happy", "closed", BOX, (832, 1216)) == [face]
    assert frame_passes("happy", "open", BOX, (832, 1216)) == [mouth]
    assert frame_passes("happy", "blink", BOX, (832, 1216)) == [eyes]
    assert frame_passes("covering_eyes", "closed", BOX, (832, 1216)) == [face, hands]
    assert frame_passes("covering_eyes", "open", BOX, (832, 1216)) == [mouth]
    assert frame_passes("covering_eyes", "blink", BOX, (832, 1216)) == [eyes]


def test_open_and_blink_inpaint_only_their_feathered_sub_mask(tmp_path):
    grey_base(tmp_path)
    masks = {}

    def record(image, mask, prompt, strength, seed):
        masks.setdefault(prompt, mask)
        return Image.new("RGB", image.size, (255, 0, 0))

    run_jobs([("sad", "open"), ("sad", "blink")], tmp_path, "white_tee", 1, BOX, record)
    by_variant = {v: m for p, m in masks.items() for v in ("open", "blink") if VARIANT_PROMPTS[v] in p}
    assert by_variant["open"].getpixel((416, 350)) == 255 and by_variant["open"].getpixel((416, 210)) == 0
    assert by_variant["open"].getpixel((300, 400)) == 0 and 0 < by_variant["open"].getpixel((336, 400)) < 255
    assert by_variant["blink"].getpixel((416, 210)) == 255 and by_variant["blink"].getpixel((416, 400)) == 0
    assert by_variant["blink"].getpixel((416, 100)) == 0 and 0 < by_variant["blink"].getpixel((288, 210)) < 255
    with Image.open(tmp_path / "sad_closed.png") as closed, Image.open(tmp_path / "sad_open.png") as opened:
        assert opened.getpixel((416, 210)) == closed.getpixel((416, 210))  # the eyes stay as in closed
        assert opened.getpixel((416, 350)) == (255, 0, 0)


def test_parse_expressions_validates_against_the_expression_prompts():
    assert parse_expressions("pouty, sad") == ["pouty", "sad"]
    for bad in ("pouty,pouting", " , "):
        with pytest.raises(argparse.ArgumentTypeError, match="choose from"):
            parse_expressions(bad)


def test_parse_face_box_needs_four_ordered_ints_inside_the_base():
    assert parse_face_box("1, 2,30,40") == (1, 2, 30, 40)
    assert parse_face_box("0,0,832,1216") == (0, 0, 832, 1216)
    for bad in ("1,2,3", "a,b,c,d", "10,2,5,40", "10,40,50,40", "-1,0,10,10", "0,0,833,100", "0,0,100,1217"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_face_box(bad)


def test_frame_seed_changes_with_the_reroll_count():
    assert frame_seed("pouty", 5, {}) == stable_seed("pouty", 5)
    assert frame_seed("pouty", 5, {"pouty": 2, "sad": 1}) == stable_seed("pouty", 5) + 2000
    assert frame_seed("happy", 5, {"pouty": 2}) == stable_seed("happy", 5)


def test_merge_rerolls_takes_the_larger_count_bumps_rerolled_names_and_reports_changes():
    assert merge_rerolls({}, {}, []) == ({}, [])
    assert merge_rerolls({"sad": 1}, {}, ["pouty"]) == ({"pouty": 1, "sad": 1}, ["pouty"])
    assert merge_rerolls({"sad": 1}, {"sad": 3, "happy": 0}, ["sad"]) == ({"sad": 4}, ["sad"])
    assert merge_rerolls({"sad": 2}, {"sad": 1}, []) == ({"sad": 2}, [])


def test_load_choice_reads_and_validates_rerolls(tmp_path):
    path = tmp_path / "choice.json"
    path.write_text(json.dumps({"outfit": "white_tee", "seed": 4}))
    assert load_choice(path) == ("white_tee", 4, {})
    path.write_text(json.dumps({"outfit": "white_tee", "seed": 4, "rerolls": {"pouty": 2}}))
    assert load_choice(path) == ("white_tee", 4, {"pouty": 2})
    for bad, match in (({"pouting": 1}, "pouting"), ({"pouty": -1}, "pouty"), ({"pouty": "x"}, "pouty"), ([1], "rerolls")):
        path.write_text(json.dumps({"outfit": "white_tee", "seed": 4, "rerolls": bad}))
        with pytest.raises(SystemExit, match=match):
            load_choice(path)


@pytest.mark.parametrize("flag", ["--only", "--reroll"])
def test_main_rejects_unknown_expression_names_before_doing_any_work(tmp_path, capsys, flag):
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "white_tee", "seed": 1}))
    argv = ["--choice", str(tmp_path / "choice.json"), "--refs", str(tmp_path), "--out", str(tmp_path / "o"), flag, "nope"]
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
    for flag in ("--choice", "--only", "--reroll", "--reset-run", "--preview-box"):
        assert flag in script.stdout


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
    assert [c["strength"] for c in calls] == [0.55, 0.55, 0.55, 0.75]  # only covering_eyes closed has a hands pass
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


class FakeTorch:
    """Stands in for torch so main()'s generation path runs without it; records generator seeds."""

    seeds: list = []

    class cuda:
        @staticmethod
        def empty_cache():
            pass

    class Generator:
        def __init__(self, device):
            pass

        def manual_seed(self, seed):
            FakeTorch.seeds.append(seed)
            return self


class FakePipe:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return type("Out", (), {"images": [Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3))]})()


def fake_generation(monkeypatch, module, kinds, calls):
    FakeTorch.seeds = []
    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(module, "load_refs", lambda folder: ["ref"])

    def fake_load(kind, lora=None, lora_scale=0.8):
        kinds.append(kind)
        return FakePipe(calls)

    monkeypatch.setattr(module, "load_pipeline", fake_load)


def frames_dir(tmp_path, *, with_base):
    out = tmp_path / "frames"
    out.mkdir()
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "pink_hoodie", "seed": 22}))
    for expr in EXPRESSION_PROMPTS:
        for variant in VARIANT_PROMPTS:
            Image.new("RGB", (64, 96)).save(out / f"{expr}_{variant}.png")
    if with_base:
        Image.new("RGB", (64, 96)).save(out / "base.png")
    return out


def expressions_argv(tmp_path, out, *extra):
    return ["--choice", str(tmp_path / "choice.json"), "--refs", str(tmp_path), "--out", str(out), *extra]


def test_base_is_the_picked_candidate_copied_and_loads_no_pipeline(tmp_path, monkeypatch):
    out = frames_dir(tmp_path, with_base=False)
    (tmp_path / "outfits").mkdir()  # the default --outfits is <out>/../outfits
    Image.new("RGB", (416, 608), (9, 8, 7)).save(tmp_path / "outfits" / "outfit_pink_hoodie_22.png")
    Image.new("RGB", (416, 608), (1, 1, 1)).save(tmp_path / "outfits" / "outfit_pink_hoodie_11.png")
    kinds = []
    fake_generation(monkeypatch, gen_expressions, kinds, [])
    gen_expressions.main(expressions_argv(tmp_path, out))
    assert kinds == []
    with Image.open(out / "base.png") as base:
        assert base.size == (832, 1216) and base.getpixel((400, 600)) == (9, 8, 7)  # the seed-22 candidate, resized
    assert [p.name for p in out.iterdir() if p.name.endswith(".tmp.png")] == []


def test_explicit_outfits_dir_is_used(tmp_path, monkeypatch):
    out = frames_dir(tmp_path, with_base=False)
    Image.new("RGB", (832, 1216), (5, 5, 5)).save(tmp_path / "elsewhere.png")
    (tmp_path / "picks").mkdir()
    (tmp_path / "elsewhere.png").replace(tmp_path / "picks" / "outfit_pink_hoodie_22.png")
    fake_generation(monkeypatch, gen_expressions, [], [])
    gen_expressions.main(expressions_argv(tmp_path, out, "--outfits", str(tmp_path / "picks")))
    with Image.open(out / "base.png") as base:
        assert base.getpixel((10, 10)) == (5, 5, 5)


def test_missing_candidate_regenerates_with_exactly_the_outfit_sheets_prompt_and_seed(tmp_path, monkeypatch):
    out = frames_dir(tmp_path, with_base=False)
    kinds, calls = [], []
    fake_generation(monkeypatch, gen_expressions, kinds, calls)
    gen_expressions.main(expressions_argv(tmp_path, out))
    assert kinds == ["txt2img"] and FakeTorch.seeds == [22]
    assert (out / "base.png").exists() and calls[0]["width"] == 832 and calls[0]["height"] == 1216

    sheet_calls = []
    fake_generation(monkeypatch, gen_outfits, [], sheet_calls)
    sheet_out = tmp_path / "sheet_out"
    sheet_out.mkdir()
    for oid in OUTFITS:
        if oid != "pink_hoodie":
            Image.new("RGB", (32, 48)).save(sheet_out / f"outfit_{oid}_22.png")
    gen_outfits.main(["--refs", str(tmp_path), "--out", str(sheet_out), "--seeds", "22"])
    assert [c["prompt"] for c in sheet_calls] == [calls[0]["prompt"]]
    assert calls[0]["negative_prompt"] == sheet_calls[0]["negative_prompt"] and FakeTorch.seeds[-1] == 22


def full_frames_dir(tmp_path, choice):
    """A frames folder with an 832x1216 grey base.png and every frame filled with (90, 90, 200)."""
    out = tmp_path / "frames"
    out.mkdir()
    (tmp_path / "choice.json").write_text(json.dumps(choice))
    grey_base(out)
    filled = Image.new("RGB", (832, 1216), (90, 90, 200))
    for expr in EXPRESSION_PROMPTS:
        for variant in VARIANT_PROMPTS:
            filled.save(out / f"{expr}_{variant}.png")
    return out


def run_json(out):
    return json.loads((out / "run.json").read_text(encoding="utf-8"))


def test_first_run_writes_run_json_and_a_resume_refuses_a_different_outfit_seed_or_face_box(tmp_path, capsys):
    out = frames_dir(tmp_path, with_base=True)
    gen_expressions.main(expressions_argv(tmp_path, out, "--face-box", "200,50,600,450"))
    assert run_json(out) == {"outfit": "pink_hoodie", "seed": 22, "face_box": [200, 50, 600, 450], "rerolls": {}}
    assert [p.name for p in out.iterdir() if ".tmp" in p.name] == []
    gen_expressions.main(expressions_argv(tmp_path, out))  # no --face-box: the recorded one is used
    assert run_json(out)["face_box"] == [200, 50, 600, 450]
    capsys.readouterr()

    for choice, extra, word in (({"outfit": "pink_hoodie", "seed": 23}, (), "seed"),
                                ({"outfit": "white_tee", "seed": 22}, (), "outfit"),
                                ({"outfit": "pink_hoodie", "seed": 22}, ("--face-box", "256,60,576,420"), "face_box")):
        (tmp_path / "choice.json").write_text(json.dumps(choice))
        with pytest.raises(SystemExit) as exc:
            gen_expressions.main(expressions_argv(tmp_path, out, *extra))
        err = capsys.readouterr().err
        assert exc.value.code == 2 and word in err and "--reset-run" in err
        assert run_json(out)["seed"] == 22 and run_json(out)["outfit"] == "pink_hoodie"

    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "white_tee", "seed": 5}))
    gen_expressions.main(expressions_argv(tmp_path, out, "--reset-run"))
    assert run_json(out) == {"outfit": "white_tee", "seed": 5, "face_box": [256, 60, 576, 420], "rerolls": {}}


def test_a_corrupt_run_json_fails_clearly(tmp_path):
    out = frames_dir(tmp_path, with_base=True)
    (out / "run.json").write_text('{"outfit": "pink_hoodie", "seed": 22, "face_box": [5, 5, 1, 1], "rerolls": {}}')
    with pytest.raises(SystemExit, match="run.json"):
        gen_expressions.main(expressions_argv(tmp_path, out))


def test_reroll_regenerates_exactly_that_expressions_three_frames_with_a_new_seed(tmp_path, monkeypatch):
    out = full_frames_dir(tmp_path, {"outfit": "pink_hoodie", "seed": 22})
    kinds, calls = [], []
    fake_generation(monkeypatch, gen_expressions, kinds, calls)
    gen_expressions.main(expressions_argv(tmp_path, out))
    assert kinds == [] and run_json(out)["rerolls"] == {}

    gen_expressions.main(expressions_argv(tmp_path, out, "--reroll", "pouty"))
    assert kinds == ["inpaint"] and len(calls) == 3  # closed (face), open (mouth), blink (eyes): one pass each
    assert FakeTorch.seeds == [stable_seed("pouty", 22) + 1000] * 3
    assert run_json(out)["rerolls"] == {"pouty": 1}
    for expr in EXPRESSION_PROMPTS:
        for variant, point in (("closed", (416, 240)), ("open", (416, 350)), ("blink", (416, 210))):
            with Image.open(out / f"{expr}_{variant}.png") as im:
                assert im.getpixel(point) == ((1, 2, 3) if expr == "pouty" else (90, 90, 200)), (expr, variant)

    gen_expressions.main(expressions_argv(tmp_path, out, "--reroll", "pouty", "--only", "sad"))
    assert run_json(out)["rerolls"] == {"pouty": 2} and FakeTorch.seeds[3:] == [stable_seed("pouty", 22) + 2000] * 3
    gen_expressions.main(expressions_argv(tmp_path, out))  # a plain resume redoes nothing
    assert len(calls) == 6 and run_json(out)["rerolls"] == {"pouty": 2}


def test_rerolls_in_choice_json_set_the_seed(tmp_path, monkeypatch):
    out = tmp_path / "frames"
    out.mkdir()
    grey_base(out)
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "pink_hoodie", "seed": 22, "rerolls": {"sad": 2}}))
    fake_generation(monkeypatch, gen_expressions, [], [])
    gen_expressions.main(expressions_argv(tmp_path, out, "--only", "sad"))
    assert FakeTorch.seeds == [stable_seed("sad", 22) + 2000] * 3 and run_json(out)["rerolls"] == {"sad": 2}


def test_preview_box_draws_the_boxes_on_base_and_exits_before_inpainting(tmp_path, monkeypatch):
    out = tmp_path / "frames"
    out.mkdir()
    grey_base(out)
    (tmp_path / "choice.json").write_text(json.dumps({"outfit": "pink_hoodie", "seed": 22}))
    kinds = []
    fake_generation(monkeypatch, gen_expressions, kinds, [])
    gen_expressions.main(expressions_argv(tmp_path, out, "--preview-box"))
    assert kinds == [] and sorted(p.name for p in out.iterdir()) == ["base.png", "base_box.png"]
    with Image.open(out / "base_box.png") as preview:
        assert preview.size == (832, 1216)
        assert preview.getpixel((256, 240)) != (128, 128, 128)  # the face box's left edge
        assert preview.getpixel((128, 600)) != (128, 128, 128)  # the hands box's left edge
        assert preview.getpixel((416, 240)) == (128, 128, 128)
    (out / "run.json").write_text(json.dumps({"outfit": "pink_hoodie", "seed": 22, "face_box": [1, 2, 3, 4], "rerolls": {}}))
    gen_expressions.main(expressions_argv(tmp_path, out, "--preview-box", "--face-box", "300,100,500,300"))  # not refused
    with Image.open(out / "base_box.png") as preview:
        assert preview.getpixel((300, 200)) != (128, 128, 128) and preview.getpixel((256, 240)) == (128, 128, 128)
    assert run_json(out)["face_box"] == [1, 2, 3, 4]


def fake_remove(img):
    img = img.convert("RGB")
    differs = ImageChops.difference(img, Image.new("RGB", img.size, (128, 128, 128))).convert("L")
    img.putalpha(differs.point(lambda v: 255 if v else 0))  # grey pixels turn transparent
    return img


def frame(color):
    im = Image.new("RGB", (832, 1216), (128, 128, 128))
    im.paste(Image.new("RGB", (200, 900), color), (316, 316))
    return im


def test_normalise_keeps_anchor_same_for_all_frames():
    a, b = normalise(fake_remove(frame((255, 0, 0)))), normalise(fake_remove(frame((0, 0, 255))))
    assert a.size == b.size == (768, 1344)
    assert a.getbbox() == b.getbbox()


def test_build_pack_writes_valid_manifest_and_lists_missing(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "pack"
    raw.mkdir()
    for name in ["neutral_closed", "neutral_open", "happy_closed"]:
        frame((255, 0, 0)).save(raw / f"{name}.png")
    missing = build_pack(raw, out, fake_remove)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["expressions"]["neutral"]["open"] == "neutral_open.png"
    assert "neutral.blink" in missing and "happy.open" in missing and len(missing) == 14 * 3 - 3
    assert Image.open(out / "neutral_closed.png").getpixel((0, 0))[3] == 0


def test_build_pack_requires_neutral_closed(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    frame((255, 0, 0)).save(raw / "happy_closed.png")
    with pytest.raises(ValueError):
        build_pack(raw, tmp_path / "pack", fake_remove)


def raw_frames(folder, names):
    folder.mkdir(exist_ok=True)
    for name in names:
        frame((255, 0, 0)).save(folder / f"{name}.png")
    return folder


def schema_errors(manifest):
    schema = json.loads((ROOT / "body" / "sprite-manifest.schema.json").read_text(encoding="utf-8"))
    return list(Draft7Validator(schema).iter_errors(manifest))


def test_manifest_is_exactly_the_schema_shape_with_no_nulls_missing_list_or_empty_expressions(tmp_path):
    out = tmp_path / "pack"
    raw = raw_frames(tmp_path / "raw", ["neutral_closed", "happy_open"])  # happy has no closed: legal, the body falls back
    build_pack(raw, out, fake_remove)
    manifest = json.loads((out / "manifest.json").read_text())
    assert schema_errors(manifest) == []
    assert set(manifest) == {"name", "version", "canvas", "crops", "expressions"}
    assert manifest["name"] == "riya-v1" and manifest["version"] == 1
    assert manifest["canvas"] == {"width": 768, "height": 1344}
    assert manifest["expressions"] == {"neutral": {"closed": "neutral_closed.png"}, "happy": {"open": "happy_open.png"}}
    assert sorted(p.name for p in out.iterdir()) == ["happy_open.png", "manifest.json", "neutral_closed.png"]


def crop_box_height(box_width, crop):
    """The body's cropBox(boxWidth, canvas, crop).height (body/src/stage-logic.js): h * boxWidth / w."""
    return crop[3] * box_width / crop[2]


def test_upper_crop_is_head_and_shoulders_through_the_normalise_transform_and_clamped(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed"])
    build_pack(raw, tmp_path / "a", fake_remove)
    crop = json.loads((tmp_path / "a" / "manifest.json").read_text())["crops"]["upper"]
    # raw box 96..736 x 6..672 (2x the face width, 0.15h above, 0.7h below) at scale 768/832, bottom offset 222
    assert crop == [88, 227, 592, 616]
    assert 0.8 <= crop[2] / crop[3] <= 1.4 and crop_box_height(320, crop) <= 400
    build_pack(raw, tmp_path / "b", fake_remove, face_override=(0, 900, 832, 1200))  # runs past the canvas
    crop = json.loads((tmp_path / "b" / "manifest.json").read_text())["crops"]["upper"]
    x, y, w, h = crop
    assert (x, w, y + h) == (0, 768, 1344) and all(isinstance(v, int) for v in crop)
    assert schema_errors(json.loads((tmp_path / "b" / "manifest.json").read_text())) == []


@pytest.mark.parametrize("box", [(256, 60, 576, 420), (300, 120, 540, 400), (200, 40, 640, 500)])
def test_upper_crop_fits_the_panel_for_plausible_face_boxes(box):
    crop = build_pack_module.upper_crop((832, 1216), box)
    assert 0.8 <= crop[2] / crop[3] <= 1.4 and crop_box_height(320, crop) <= 400


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_bodys_crop_box_for_the_default_crop_is_at_most_400_tall():
    crop = build_pack_module.upper_crop((832, 1216), BOX)
    script = (
        "const { cropBox } = require(require('path').resolve('body/src/stage-logic.js'));"
        "console.log(cropBox(320, { width: 768, height: 1344 }, JSON.parse(process.argv[1])).height);"
    )
    out = subprocess.run(["node", "-e", script, json.dumps(crop)], check=True, cwd=ROOT, capture_output=True, text=True)
    assert float(out.stdout) == pytest.approx(crop_box_height(320, crop)) and float(out.stdout) <= 400


def test_build_pack_takes_the_face_box_from_run_json_unless_overridden(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed"])
    (raw / "run.json").write_text(json.dumps({"outfit": "white_tee", "seed": 1, "face_box": [300, 120, 540, 400],
                                              "rerolls": {}}))
    build_pack(raw, tmp_path / "a", fake_remove)
    crop = json.loads((tmp_path / "a" / "manifest.json").read_text())["crops"]["upper"]
    assert crop == build_pack_module.upper_crop((832, 1216), (300, 120, 540, 400))
    build_pack(raw, tmp_path / "b", fake_remove, face_override=BOX)
    crop = json.loads((tmp_path / "b" / "manifest.json").read_text())["crops"]["upper"]
    assert crop == build_pack_module.upper_crop((832, 1216), BOX)
    assert read_run(raw)["face_box"] == [300, 120, 540, 400] and read_run(tmp_path / "a") is None


def test_every_frame_has_the_canvas_size_and_the_same_bottom_centred_anchor(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed", "sad_blink"])
    build_pack(raw, tmp_path / "pack", fake_remove)
    boxes = []
    for name in ("neutral_closed", "sad_blink"):
        with Image.open(tmp_path / "pack" / f"{name}.png") as im:
            assert im.size == (768, 1344) and im.mode == "RGBA"
            boxes.append(im.getbbox())
    # the 832x1216 frame scales to 768x1122 and sits at y=222..1344; the art (y 316..1216) ends on the canvas bottom
    assert boxes[0] == boxes[1] and boxes[0][3] == 1344


def test_rebuild_overwrites_and_removes_frames_no_longer_in_raw(tmp_path):
    out = tmp_path / "pack"
    raw = raw_frames(tmp_path / "raw", ["neutral_closed", "sad_open"])
    build_pack(raw, out, fake_remove)
    (out / "notes.txt").write_text("keep me")
    (out / "stale.tmp.png").write_bytes(b"half")
    (raw / "sad_open.png").unlink()
    build_pack(raw, out, fake_remove)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "sad" not in manifest["expressions"]
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "neutral_closed.png", "notes.txt"]


def test_build_pack_ignores_non_frame_pngs_in_raw(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed", "base", "review_sheet", "bogus_closed", "happy_wink"])
    build_pack(raw, tmp_path / "pack", fake_remove)
    assert sorted(p.name for p in (tmp_path / "pack").iterdir()) == ["manifest.json", "neutral_closed.png"]


def test_mismatched_frame_size_fails_clearly_before_writing_anything(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed"])
    Image.new("RGB", (400, 600), (128, 128, 128)).save(raw / "happy_closed.png")
    with pytest.raises(ValueError, match=r"happy_closed\.png.*400x600"):
        build_pack(raw, tmp_path / "pack", fake_remove)
    assert not (tmp_path / "pack").exists()


def test_build_pack_and_default_remover_import_without_rembg():
    code = (
        "import sys, colab.sprites.build_pack as b;"
        "assert callable(b.default_remover) and 'rembg' not in sys.modules and 'torch' not in sys.modules"
        " and 'diffusers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT)


def test_build_pack_script_help_and_cli_main(tmp_path, monkeypatch, capsys):
    script = subprocess.run([sys.executable, "colab/sprites/build_pack.py", "--help"], check=True, cwd=ROOT,
                            capture_output=True, text=True)
    assert "--raw" in script.stdout and "--out" in script.stdout
    raw = raw_frames(tmp_path / "raw", ["neutral_closed"])
    monkeypatch.setattr(build_pack_module, "default_remover", lambda: fake_remove)
    build_pack_module.main(["--raw", str(raw), "--out", str(tmp_path / "pack")])
    printed = capsys.readouterr().out
    assert "neutral.open" in printed and "happy.closed" in printed and (tmp_path / "pack" / "manifest.json").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_the_body_accepts_the_built_pack(tmp_path):
    raw = raw_frames(tmp_path / "raw", ["neutral_closed", "neutral_open", "neutral_blink", "happy_closed", "happy_open"])
    build_pack(raw, tmp_path / "pack", fake_remove)
    script = (
        "const fs = require('fs'), path = require('path');"
        "const { validateManifest } = require(path.resolve('body/src/sprite-logic.js'));"
        "const dir = process.argv[1], expressions = JSON.parse(process.argv[2]);"
        "const m = JSON.parse(fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8'));"
        "const warnings = validateManifest(m, expressions);"
        "for (const e of Object.values(m.expressions)) for (const f of Object.values(e)) fs.accessSync(path.join(dir, f));"
        "console.log(JSON.stringify(warnings));"
    )
    out = subprocess.run(["node", "-e", script, str(tmp_path / "pack"), json.dumps(protocol_expressions())], check=True,
                         cwd=ROOT, capture_output=True, text=True)
    warnings = json.loads(out.stdout)
    assert "missing happy.blink" in warnings and "missing neutral.closed" not in warnings and len(warnings) == 14 * 3 - 5
