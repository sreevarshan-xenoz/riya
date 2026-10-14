"""Voice data: affect-labelled dialogue for affect-conditioned SFT (Notion page 23, plan step 1).

Scripts: `python -m jarvis.voice.build_dataset`, `python -m jarvis.voice.synth_prompts`.
"""
from .affect_map import AFFECT_NAMES, ALL_LABELS, AffectMapper
from .chat import format_row, render_chat

__all__ = ["AFFECT_NAMES", "ALL_LABELS", "AffectMapper", "format_row", "render_chat"]
