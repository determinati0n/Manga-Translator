# -*- coding: utf-8 -*-
"""
Manga Translator V10.4
"""

import os
import re
import io
import argparse
import time
from pathlib import Path

# Paddle/PIR compatibility for the CPU setup used by this project.
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_PIR", "1")

import cv2
import fitz
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


# ---------------- CONFIG ----------------

RENDER_SCALE = 2.0

# Translation
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "rus_Cyrl"

# Optional local dialogue editor (disabled by default while NLLB is being validated)
QWEN_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
QWEN_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
AI_ROOT = Path(os.environ.get("MANGA_AI_DIR", r"D:\AI"))
WORK_ROOT = Path(os.environ.get("MANGA_TRANSLATOR_HOME", r"D:\manhva"))
QWEN_DIR = str(AI_ROOT / "models" / "qwen")
QWEN_CONTEXT = 2048
QWEN_THREADS = 6
QWEN_MAX_TOKENS = 120
QWEN_TEMPERATURE = 0.05
USE_QWEN_EDITOR = False

# Bubble segmentation model
BUBBLE_REPO = "huyvux3005/manga109-segmentation-bubble"
BUBBLE_FILE = "weights/best.pt"
BUBBLE_IMGSZ = 1600
BUBBLE_CONF = 0.25
BUBBLE_IOU = 0.50

# OCR
OCR_LANG = "en"
MIN_OCR_CONF = 0.40
MIN_TEXT_LEN = 2
OCR_UPSCALE = 2.0
OCR_DEBUG = True
OCR_DEBUG_DIR = str(WORK_ROOT / "ocr_debug")

# Inner margin inside a speech bubble.
BUBBLE_MARGIN_X = 0.10
BUBBLE_MARGIN_Y = 0.10

LOG_FILE = str(WORK_ROOT / "translation_v10.log")

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\Arial.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\Tahoma.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
]


# ---------------- HELPERS ----------------

def log(msg: str):
    print(msg)
    try:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def clean_ocr_text(text: str) -> str:
    text = str(text or "").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("|", "I")
    text = re.sub(r"^[\W_]+$", "", text)
    return text


def clean_translation(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip("“”\"' ")
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    text = re.sub(r"([!?]){3,}", r"\1", text)
    return text


def find_font():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def bbox_from_poly(poly):
    arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    return [
        int(np.min(arr[:, 0])),
        int(np.min(arr[:, 1])),
        int(np.max(arr[:, 0])),
        int(np.max(arr[:, 1])),
    ]


def clip_box(box, width, height):
    x1, y1, x2, y2 = [int(v) for v in box]
    return [
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    ]


def useful_text(text, conf):
    if conf < MIN_OCR_CONF:
        return False
    if len(text.strip()) < MIN_TEXT_LEN:
        return False
    return bool(re.search(r"[A-Za-z0-9]", text))


# ---------------- MANGA OCR HELPERS ----------------

def normalize_ocr_candidate(text: str) -> str:
    text = clean_ocr_text(text)
    # OCR commonly confuses these in stylized English text.
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"\s+([,.!?…])", r"\1", text)
    text = re.sub(r"([!?]){3,}", r"\1", text)
    return text


def ocr_candidate_score(items):
    if not items:
        return -1e9

    text = " ".join(x["text"] for x in items)
    letters = len(re.findall(r"[A-Za-z]", text))
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    avg_conf = float(np.mean([x["confidence"] for x in items]))
    bad = len(re.findall(r"[^A-Za-z0-9\s,.!?…'~\\-]", text))

    # Prefer confident English-looking text, but don't simply prefer the
    # longest result: over-detection of bubble borders can create junk.
    score = avg_conf * 100.0
    score += min(letters, 80) * 0.25
    score += min(len(words), 15) * 1.0
    score -= bad * 2.0
    if letters < 2:
        score -= 30
    return score


def make_ocr_variants(crop: Image.Image):
    arr = np.asarray(crop.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Dynamic upscale: never create a huge image that PaddleOCR immediately
    # has to shrink again.
    max_side_for_ocr = 3600.0
    base_scale = float(OCR_UPSCALE)
    longest = max(arr.shape[0], arr.shape[1])
    scale = min(base_scale, max_side_for_ocr / max(1.0, longest))
    scale = max(1.0, scale)

    up_rgb = cv2.resize(
        arr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    up = cv2.cvtColor(up_rgb, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(up)

    otsu = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Return the effective scale alongside each image. The OCR code below
    # uses it when mapping boxes back to page coordinates.
    return [
        ("rgb", Image.fromarray(up_rgb), scale),
        ("gray", Image.fromarray(up), scale),
        ("contrast", Image.fromarray(enhanced), scale),
        ("otsu", Image.fromarray(otsu), scale),
        ("adaptive", Image.fromarray(adaptive), scale),
    ]

def find_text_bands(image: Image.Image):
    """
    Estimate horizontal text lines from dark-pixel projection.

    This is deliberately conservative. If no reliable bands are found,
    normal PaddleOCR detection is used instead.
    """
    arr = np.asarray(image.convert("L"))
    h, w = arr.shape

    if h < 20 or w < 20:
        return []

    # Ignore a small border area so the bubble outline does not become a
    # fake text line.
    y0 = max(1, int(h * 0.05))
    y1 = min(h - 1, int(h * 0.95))
    work = arr[y0:y1]

    dark = (work < 185).astype(np.uint8)

    # Remove isolated noise and connect nearby characters within a line.
    kernel_w = max(3, int(w * 0.025))
    kernel_w = min(kernel_w, 31)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_w, 2),
    )
    joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

    row_score = joined.sum(axis=1)
    threshold = max(2, int(w * 0.012))

    active = row_score >= threshold

    bands = []
    start = None
    for i, flag in enumerate(active):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 2:
                bands.append((start + y0, i + y0))
            start = None

    if start is not None and len(active) - start >= 2:
        bands.append((start + y0, len(active) + y0))

    # Merge bands separated by tiny gaps.
    merged = []
    gap_limit = max(2, int(h * 0.018))
    for a, b in bands:
        if merged and a - merged[-1][1] <= gap_limit:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    # Reject lines that are implausibly tall or tiny.
    clean = []
    for a, b in merged:
        bh = b - a
        if 4 <= bh <= max(12, int(h * 0.35)):
            clean.append((max(0, a - 3), min(h, b + 3)))

    # A normal speech bubble generally has only a few text lines.
    if not (1 <= len(clean) <= 6):
        return []

    return clean


def ocr_image(ocr, image: Image.Image, x_offset=0, y_offset=0):
    """
    OCR a single image and map its coordinates back to the bubble crop.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        temp_path = tmp.name

    try:
        image.save(temp_path)
        result = ocr.predict(temp_path)
        raw = parse_paddle_result(result)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    items = []
    for text, conf, box in raw:
        text = normalize_ocr_candidate(text)
        if not useful_text(text, conf):
            continue

        x1, y1, x2, y2 = box
        items.append({
            "text": text,
            "confidence": float(conf),
            "box": [
                int(x1 + x_offset),
                int(y1 + y_offset),
                int(x2 + x_offset),
                int(y2 + y_offset),
            ],
        })
    return items


def ocr_crop_enhanced(ocr, crop: Image.Image, offset_x: int, offset_y: int):
    """
    Conservative OCR for manga bubbles.

    Fast path:
      1) RGB upscale
      2) grayscale only if RGB is weak
      3) contrast/threshold only as fallbacks

    No horizontal-band slicing: that previously created giant strips and
    caused PaddleOCR max_side_limit warnings.
    """
    variants = make_ocr_variants(crop)

    def run_variant(variant, scale):
        items = ocr_image(ocr, variant, 0, 0)

        for item in items:
            item["box"] = [
                int(item["box"][0] / scale + offset_x),
                int(item["box"][1] / scale + offset_y),
                int(item["box"][2] / scale + offset_x),
                int(item["box"][3] / scale + offset_y),
            ]
        return items

    candidates = []

    # RGB is often best for anti-aliased/stylized lettering.
    rgb_items = run_variant(variants[0][1], variants[0][2])
    if rgb_items:
        candidates.append(("rgb", rgb_items))

    best_score = ocr_candidate_score(rgb_items)

    # Grayscale confirmation/fallback.
    if best_score < 60.0:
        gray_items = run_variant(variants[1][1], variants[1][2])
        if gray_items:
            candidates.append(("gray", gray_items))
            best_score = max(best_score, ocr_candidate_score(gray_items))

    # Expensive variants only if necessary.
    if best_score < 45.0:
        for name, variant, scale in variants[2:]:
            items = run_variant(variant, scale)
            if items:
                candidates.append((name, items))
            best_score = max(
                best_score,
                ocr_candidate_score(items),
            )
            if best_score >= 60.0:
                break

    if not candidates:
        return []

    candidates.sort(
        key=lambda pair: ocr_candidate_score(pair[1]),
        reverse=True,
    )
    _, selected = candidates[0]

    selected = deduplicate_ocr_items(selected)
    selected.sort(key=lambda x: (x["box"][1], x["box"][0]))
    return selected

def deduplicate_ocr_items(items):
    if len(items) <= 1:
        return items

    kept = []
    for item in sorted(
        items,
        key=lambda x: (
            -(x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]),
            -x["confidence"],
        ),
    ):
        duplicate = False
        for other in kept:
            ax1, ay1, ax2, ay2 = item["box"]
            bx1, by1, bx2, by2 = other["box"]

            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area = max(1, (ax2 - ax1) * (ay2 - ay1))
            if inter / area > 0.65:
                duplicate = True
                break

        if not duplicate:
            kept.append(item)

    return kept


def group_ocr_lines(items):
    """
    Preserve the visual line order. If OCR has already returned one combined
    line, do not try to invent a new order from words.
    """
    if not items:
        return []

    items = sorted(
        items,
        key=lambda x: (
            x["box"][1],
            x["box"][0],
        ),
    )

    heights = [
        max(1, x["box"][3] - x["box"][1])
        for x in items
    ]
    median_h = float(np.median(heights))

    # More conservative than V7: boxes on different visual lines should not
    # be merged merely because their x coordinates are close.
    max_gap = max(4.0, median_h * 0.75)

    groups = []

    for item in items:
        x1, y1, x2, y2 = item["box"]
        placed = False

        for group in reversed(groups[-3:]):
            gx1 = min(x["box"][0] for x in group)
            gy1 = min(x["box"][1] for x in group)
            gx2 = max(x["box"][2] for x in group)
            gy2 = max(x["box"][3] for x in group)

            vertical_gap = 0
            if y1 >= gy2:
                vertical_gap = y1 - gy2
            elif gy1 >= y2:
                vertical_gap = gy1 - y2

            overlap_y = max(
                0,
                min(y2, gy2) - max(y1, gy1),
            )
            denom = max(
                1,
                min(y2 - y1, gy2 - gy1),
            )
            overlap_y_ratio = overlap_y / denom

            if (
                vertical_gap <= max_gap
                and overlap_y_ratio >= 0.35
            ):
                group.append(item)
                placed = True
                break

        if not placed:
            groups.append([item])

    result = []
    for group in groups:
        group.sort(key=lambda x: x["box"][0])
        text = " ".join(x["text"] for x in group)
        result.append({
            "items": group,
            "text": normalize_ocr_candidate(text),
        })

    return result



# ---------------- NLLB ----------------

class NLLBTranslator:
    def __init__(self):
        log("Загрузка NLLB-200 600M...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            TRANSLATION_MODEL,
            src_lang=SRC_LANG,
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

        self.target_id = self.tokenizer.convert_tokens_to_ids(TGT_LANG)
        log(f"NLLB device: {self.device}")

    @torch.inference_mode()
    def translate(self, text: str) -> str:
        text = clean_ocr_text(text)
        if not text:
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            forced_bos_token_id=self.target_id,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

        return clean_translation(
            self.tokenizer.decode(out[0], skip_special_tokens=True)
        )


# ---------------- PADDLE OCR ----------------

def parse_paddle_result(result):
    """
    Supports the current PaddleOCR JSON-like result objects and
    the older list-based result format.
    """
    items = []

    for res in result:
        data = None

        if hasattr(res, "json"):
            try:
                data = res.json
                if callable(data):
                    data = data()
            except Exception:
                data = None

        if isinstance(data, str):
            try:
                import json
                data = json.loads(data)
            except Exception:
                data = None

        if isinstance(data, dict):
            data = [data]

        if isinstance(data, list):
            for block in data:
                if not isinstance(block, dict):
                    continue

                block = block.get("res", block)

                texts = (
                    block.get("rec_texts")
                    or block.get("texts")
                    or block.get("text")
                    or []
                )
                scores = (
                    block.get("rec_scores")
                    or block.get("scores")
                    or block.get("score")
                    or []
                )
                boxes = (
                    block.get("rec_boxes")
                    or block.get("boxes")
                    or block.get("dt_polys")
                    or block.get("polys")
                    or []
                )

                if isinstance(texts, str):
                    texts = [texts]
                if isinstance(scores, (float, int)):
                    scores = [scores]
                if isinstance(boxes, np.ndarray):
                    boxes = boxes.tolist()

                for i, text in enumerate(texts):
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if i < len(boxes):
                        items.append(
                            (
                                text,
                                score,
                                bbox_from_poly(boxes[i]),
                            )
                        )

    # Old PaddleOCR format fallback.
    if not items:
        try:
            for res in result:
                if isinstance(res, list):
                    for line in res:
                        if (
                            isinstance(line, (list, tuple))
                            and len(line) >= 2
                            and isinstance(line[1], (list, tuple))
                        ):
                            poly = line[0]
                            rec = line[1]
                            if len(rec) >= 2:
                                items.append(
                                    (
                                        rec[0],
                                        float(rec[1]),
                                        bbox_from_poly(poly),
                                    )
                                )
        except Exception:
            pass

    return items


def ocr_crop(ocr, crop: Image.Image, offset_x: int, offset_y: int):
    with __import__("tempfile").NamedTemporaryFile(
        suffix=".png", delete=False
    ) as tmp:
        temp_path = tmp.name

    try:
        crop.save(temp_path)
        result = ocr.predict(temp_path)
        raw = parse_paddle_result(result)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    items = []

    for text, conf, box in raw:
        text = clean_ocr_text(text)
        if not useful_text(text, conf):
            continue

        x1, y1, x2, y2 = box
        items.append(
            {
                "text": text,
                "confidence": conf,
                "box": [
                    x1 + offset_x,
                    y1 + offset_y,
                    x2 + offset_x,
                    y2 + offset_y,
                ],
            }
        )

    return items


# ---------------- QWEN DIALOGUE EDITOR ----------------

class QwenEditor:
    """
    Conservative post-editor.

    NLLB supplies the semantic translation. Qwen may only make a minimal
    natural-Russian correction. If the model changes the meaning too much,
    the NLLB draft is kept.
    """

    def __init__(self):
        if Llama is None:
            raise RuntimeError(
                "Не установлен llama-cpp-python. "
                "Установите CPU wheel командой:\n"
                "python -m pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu"
            )

        Path(QWEN_DIR).mkdir(parents=True, exist_ok=True)

        log("Загрузка Qwen2.5-3B-Instruct Q4_K_M...")
        model_path = hf_hub_download(
            repo_id=QWEN_REPO,
            filename=QWEN_FILE,
            local_dir=QWEN_DIR,
        )
        log(f"Qwen model: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=QWEN_CONTEXT,
            n_threads=QWEN_THREADS,
            n_gpu_layers=0,
            verbose=False,
        )

        log("Qwen консервативный редактор готов.")

    def edit(self, source: str, draft: str) -> str:
        source = clean_ocr_text(source)
        draft = clean_translation(draft)

        if not source:
            return draft
        if not draft:
            return draft

        system = (
            "Ты редактор перевода английской манги на русский. "
            "NLLB уже сделал перевод, и его смысл нужно считать правильным. "
            "Твоя задача — только исправить явную русскую кальку, если она "
            "звучит неестественно. Если черновик нормальный, верни его БЕЗ ИЗМЕНЕНИЙ. "
            "НИКОГДА не добавляй новый смысл, факт, эмоцию, срочность, причину, "
            "угрозу, приказ, ласковость или грубость. "
            "Не меняй отрицание, время, модальность или местоимения. "
            "Не смягчай ругань. "
            "Минимальная естественная перестановка слов разрешена. "
        )

        user = (
            f"ОРИГИНАЛ:\n{source}\n\n"
            f"ЧЕРНОВОЙ ПЕРЕВОД NLLB:\n{draft}\n\n"
            "Исправь только явную кальку. Если исправлять нечего — "
            "верни черновик дословно."
        )

        try:
            result = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.05,
                top_p=0.85,
                max_tokens=QWEN_MAX_TOKENS,
            )

            edited = clean_translation(
                result["choices"][0]["message"]["content"]
            )

            if not edited:
                return draft

            # Reject explanations/meta answers.
            low = edited.lower()
            if any(
                low.startswith(prefix)
                for prefix in (
                    "перевод:",
                    "translation:",
                    "вот перевод",
                    "исправленный перевод:",
                    "corrected translation:",
                )
            ):
                return draft

            # Conservative length guard.
            if len(edited) > max(300, len(draft) * 2.0 + 30):
                return draft

            # Very short source/draft lines should not suddenly become much
            # longer due to hallucinated context.
            if len(source) <= 30 and len(edited) > max(
                len(draft) * 1.7,
                len(draft) + 16,
            ):
                return draft

            return edited

        except Exception as exc:
            log(f"Qwen editor error: {exc}")
            return draft


# ---------------- BUBBLE DETECTOR ----------------

class BubbleDetector:
    def __init__(self):
        log("Загрузка YOLO11n-seg модели пузырей...")
        model_path = hf_hub_download(
            repo_id=BUBBLE_REPO,
            filename=BUBBLE_FILE,
            local_dir=str(AI_ROOT / "models" / "bubble"),
        )
        log(f"Bubble model: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, image: Image.Image):
        rgb = np.asarray(image)
        result = self.model.predict(
            source=rgb,
            imgsz=BUBBLE_IMGSZ,
            conf=BUBBLE_CONF,
            iou=BUBBLE_IOU,
            device="cpu",
            verbose=False,
        )[0]

        bubbles = []

        if result.masks is None or result.boxes is None:
            return bubbles

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()

        H, W = rgb.shape[:2]

        for i, mask_small in enumerate(masks):
            mask = cv2.resize(
                mask_small.astype(np.float32),
                (W, H),
                interpolation=cv2.INTER_NEAREST,
            )
            mask = (mask >= 0.5).astype(np.uint8)

            # Remove tiny accidental regions.
            area = int(mask.sum())
            if area < 300:
                continue

            x1, y1, x2, y2 = [int(v) for v in boxes[i]]
            x1, y1, x2, y2 = clip_box([x1, y1, x2, y2], W, H)

            bubbles.append(
                {
                    "mask": mask,
                    "box": [x1, y1, x2, y2],
                    "confidence": float(confs[i]),
                    "area": area,
                }
            )

        # Reading order: top to bottom, then left to right.
        bubbles.sort(key=lambda b: (b["box"][1], b["box"][0]))
        return bubbles


# ---------------- OCR + TEXT GROUPING ----------------

def mask_crop(image: Image.Image, mask, box):
    x1, y1, x2, y2 = box
    crop = np.asarray(image)[y1:y2, x1:x2].copy()
    m = mask[y1:y2, x1:x2]

    # Keep the actual bubble, but make outside pixels white so OCR does not
    # see surrounding artwork.
    crop[m == 0] = 255

    return Image.fromarray(crop)




# ---------------- RENDERING ----------------

def wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return []

    lines = []
    current = words[0]

    for word in words[1:]:
        trial = current + " " + word
        b = draw.textbbox((0, 0), trial, font=font)

        if b[2] - b[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def block_size(draw, lines, font, spacing):
    if not lines:
        return 0, 0

    widths = []
    heights = []

    for line in lines:
        b = draw.textbbox((0, 0), line, font=font)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])

    return (
        max(widths),
        sum(heights) + spacing * (len(lines) - 1),
    )


def choose_font(draw, text, region, font_path):
    x1, y1, x2, y2 = region
    max_w = max(20, x2 - x1)
    max_h = max(16, y2 - y1)

    start = max(10, int(max_h * 0.34))
    minimum = max(8, int(max_h * 0.10))

    for size in range(start, minimum - 1, -1):
        font = ImageFont.truetype(font_path, size=size)
        spacing = max(1, int(size * 0.10))
        lines = wrap_text(draw, text, font, max_w * 0.94)
        tw, th = block_size(draw, lines, font, spacing)

        if tw <= max_w * 0.94 and th <= max_h * 0.90:
            return font, lines, spacing, tw, th

    font = ImageFont.truetype(font_path, size=minimum)
    spacing = max(1, int(minimum * 0.08))
    lines = wrap_text(draw, text, font, max_w * 0.94)
    tw, th = block_size(draw, lines, font, spacing)

    return font, lines, spacing, tw, th


def bubble_text_region(bubble):
    x1, y1, x2, y2 = bubble["box"]
    w = max(20, x2 - x1)
    h = max(20, y2 - y1)

    mx = max(8, int(w * BUBBLE_MARGIN_X))
    my = max(8, int(h * BUBBLE_MARGIN_Y))

    return [
        x1 + mx,
        y1 + my,
        max(x1 + mx + 20, x2 - mx),
        max(y1 + my + 20, y2 - my),
    ]


def erase_ocr_from_bubble(image: Image.Image, bubble, ocr_items):
    """
    Paint only OCR text areas white, and clip the mask to the YOLO bubble mask.
    """
    rgb = np.asarray(image).copy()
    H, W = rgb.shape[:2]

    text_mask = np.zeros((H, W), dtype=np.uint8)

    for item in ocr_items:
        x1, y1, x2, y2 = item["box"]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        px = max(3, int(bw * 0.20))
        py = max(3, int(bh * 0.25))

        cv2.rectangle(
            text_mask,
            (max(0, x1 - px), max(0, y1 - py)),
            (min(W - 1, x2 + px), min(H - 1, y2 + py)),
            255,
            -1,
        )

    # Absolutely clip the erase region to the detected speech bubble.
    text_mask = cv2.bitwise_and(
        text_mask,
        (bubble["mask"] * 255).astype(np.uint8),
    )

    rgb[text_mask > 0] = (255, 255, 255)
    image.paste(Image.fromarray(rgb), (0, 0))


def draw_translation(image, bubble, translation, font_path):
    draw = ImageDraw.Draw(image)
    region = bubble_text_region(bubble)

    font, lines, spacing, tw, th = choose_font(
        draw,
        translation,
        region,
        font_path,
    )

    x1, y1, x2, y2 = region
    width = x2 - x1
    height = y2 - y1

    ty = y1 + max(0, (height - th) / 2)

    for line in lines:
        b = draw.textbbox((0, 0), line, font=font)
        lw = b[2] - b[0]
        lh = b[3] - b[1]

        draw.text(
            (
                x1 + (width - lw) / 2,
                ty,
            ),
            line,
            fill="black",
            font=font,
        )

        ty += lh + spacing


# ---------------- PDF ----------------

def parse_pages(spec, page_count):
    if not spec or spec.lower() == "all":
        return list(range(page_count))

    result = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a)
            b = int(b)

            for p in range(min(a, b), max(a, b) + 1):
                if 1 <= p <= page_count:
                    result.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= page_count:
                result.add(p - 1)

    return sorted(result)


def process_pdf(input_pdf, output_pdf, pages_spec):
    font_path = find_font()
    if not font_path:
        raise RuntimeError(
            "Не найден Windows-шрифт с поддержкой кириллицы."
        )

    bubble_detector = BubbleDetector()

    log("Инициализация PaddleOCR English...")
    ocr = PaddleOCR(
        lang="en",
        enable_mkldnn=False,
        device="cpu",
        engine="paddle",
    )

    translator = NLLBTranslator()
    editor = QwenEditor() if USE_QWEN_EDITOR else None
    if not USE_QWEN_EDITOR:
        log("Qwen отключён: используем проверенный NLLB как финальный перевод.")

    doc = fitz.open(input_pdf)
    out = fitz.open()

    pages = parse_pages(pages_spec, len(doc))

    log(f"Страниц в PDF: {len(doc)}")
    log(f"Обрабатываем: {[p + 1 for p in pages]}")

    for page_index in range(len(doc)):
        src_page = doc[page_index]

        if page_index not in pages:
            out.insert_pdf(
                doc,
                from_page=page_index,
                to_page=page_index,
            )
            continue

        page_started = time.perf_counter()
        log(f"\n--- PAGE {page_index + 1} ---")

        pix = src_page.get_pixmap(
            matrix=fitz.Matrix(RENDER_SCALE, RENDER_SCALE),
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples,
        )

        bubbles = bubble_detector.detect(image)
        log(f"Bubble instances: {len(bubbles)}")

        for bi, bubble in enumerate(bubbles, 1):
            x1, y1, x2, y2 = bubble["box"]

            crop = mask_crop(
                image,
                bubble["mask"],
                bubble["box"],
            )

            ocr_items = ocr_crop_enhanced(
                ocr,
                crop,
                x1,
                y1,
            )

            if not ocr_items:
                log(f"Bubble {bi}: OCR empty")
                continue

            groups = group_ocr_lines(ocr_items)
            if not groups:
                log(f"Bubble {bi}: OCR groups empty")
                continue

            # One bubble is treated as one dialogue block. If Paddle detects
            # truly separate horizontal text blocks, they are joined in
            # visual order before translation.
            source_parts = [g["text"] for g in groups if g["text"]]
            source = normalize_ocr_candidate(" ".join(source_parts))

            if not source:
                continue

            log(f"Bubble {bi} | OCR: {source}")

            t_translate = time.perf_counter()

            draft = translator.translate(source)
            log(f"Bubble {bi} | NLLB: {draft}")

            if editor is not None:
                final = editor.edit(source, draft)
            else:
                final = draft

            final = clean_translation(final)

            log(
                f"Bubble {bi} | FINAL: {final} "
                f"(translate {time.perf_counter() - t_translate:.2f}s)"
            )

            if not final:
                continue

            # Erase all OCR text in this bubble in one pass.
            all_items = []
            for group in groups:
                all_items.extend(group["items"])

            erase_ocr_from_bubble(
                image,
                bubble,
                all_items,
            )

            draw_translation(
                image,
                bubble,
                final,
                font_path,
            )

        log(f"PAGE {page_index + 1} TIME: {time.perf_counter() - page_started:.2f}s")

        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)

        new_page = out.new_page(
            width=image.width / RENDER_SCALE,
            height=image.height / RENDER_SCALE,
        )
        new_page.insert_image(
            new_page.rect,
            stream=buf.getvalue(),
        )

    out.save(
        output_pdf,
        garbage=4,
        deflate=True,
    )
    out.close()
    doc.close()

    log(f"\nГотово: {output_pdf}")



def main():
    parser = argparse.ArgumentParser(
        description="Manga Translator V10.4: English -> Russian"
    )

    parser.add_argument(
        "--input",
        default=str(WORK_ROOT / "input.pdf"),
    )
    parser.add_argument(
        "--output",
        default=str(WORK_ROOT / "output_v10_4.pdf"),
    )
    parser.add_argument(
        "--pages",
        default="all",
        help='Например: "4", "4-10", "4,6,10" или "all"',
    )

    args = parser.parse_args()

    process_pdf(
        args.input,
        args.output,
        args.pages,
    )


if __name__ == "__main__":
    main()
