"""Prompt set for Riya's sprite pack: shared by the outfit sheet, expression and variant generators."""

BASE_PROMPT = (
    "1girl, solo, alya, long silver hair, blue eyes, red hair ribbon, upper body to thighs, standing, "
    "facing viewer, simple grey background, anime style, masterpiece, best quality"
)
NEGATIVE = "lowres, bad anatomy, bad hands, extra fingers, text, watermark, signature, multiple girls, cropped head, nsfw"

OUTFITS = {
    "sweater_plaid": "black off-shoulder ribbed sweater, grey plaid skirt",
    "pink_hoodie": "oversized pale pink hoodie, black leggings",
    "jacket_skirt": "black and red track jacket worn off shoulders, white sleeveless top, pink skirt, black thighhighs",
    "white_tee": "white loose t-shirt, denim shorts",
}

EXPRESSION_PROMPTS = {
    "neutral": "calm gentle expression",
    "happy": "happy smile",
    "excited": "excited, sparkling eyes, wide smile",
    "missed_you": "happy tears, teary eyes, relieved smile, blush",
    "pouty": "pouting, puffed cheeks, looking away, blush",
    "flustered": "flustered, heavy blush, embarrassed, wavy mouth",
    "teasing": "smug, teasing smirk, half-closed eyes",
    "worried": "worried, furrowed brows, concerned",
    "sad": "sad, downcast eyes, frown",
    "sleepy": "sleepy, drowsy half-closed eyes, yawning",
    "curious": "curious, head tilt, raised eyebrow",
    "thinking": "thinking, looking up, finger on chin",
    "surprised": "surprised, wide eyes, small open mouth",
    "covering_eyes": "covering eyes with both hands, peeking, blush",
}

VARIANT_PROMPTS = {"closed": "closed mouth", "open": "open mouth, talking", "blink": "closed eyes, closed mouth"}
