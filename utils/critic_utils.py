from PIL import Image
import io
import json

def crop_image(image, x: float, y: float, width: float, height: float):
    """Crop an image using normalized [0..1] coordinates, returning a JPEG PIL Image."""
    w, h = image.size
    x  = min(max(0, x), 1)
    y  = min(max(0, y), 1)
    x2 = min(max(0, x + width), 1)
    y2 = min(max(0, y + height), 1)
    cropped_img = image.crop((x * w, y * h, x2 * w, y2 * h))
    buffer = io.BytesIO()
    cropped_img.save(buffer, format="JPEG")
    buffer.seek(0)
    return Image.open(buffer)


def zoom_in_image_by_bbox(image, box, padding=0.01):
    """Crop image to a bounding box [x, y, w, h] with optional padding."""
    assert padding >= 0.01, "The padding should be at least 0.01"
    x, y, w, h = box
    return crop_image(image, x - padding, y - padding, w + 2 * padding, h + 2 * padding)


def parse_inch_string(inch_str: str) -> float:
    """
    Convert a string like '12.0 Inches' into a float (12.0).
    """
    return float(inch_str.replace(" Inches", "").strip())

def convert_pptx_bboxes_to_image_space(bbox_dict, slide_width_in, slide_height_in):
    """Convert PPTX bounding boxes (inch strings) to normalized [x, y, w, h] coords."""
    result = {}
    for label, box in bbox_dict.items():
        result[label] = [
            parse_inch_string(box['left'])   / slide_width_in,
            parse_inch_string(box['top'])    / slide_height_in,
            parse_inch_string(box['width'])  / slide_width_in,
            parse_inch_string(box['height']) / slide_height_in,
        ]
    return result

def convert_pptx_bboxes_json_to_image_json(bbox_json_str, slide_width_in, slide_height_in):
    """Convert bounding boxes (in inches) from a JSON string/dict to normalized image coords [0..1]."""
    bbox_dict = json.loads(bbox_json_str) if isinstance(bbox_json_str, str) else bbox_json_str
    normalized_bboxes = {}
    for label, box in bbox_dict.items():
        left_in   = parse_inch_string(box['left'])
        top_in    = parse_inch_string(box['top'])
        width_in  = parse_inch_string(box['width'])
        height_in = parse_inch_string(box['height'])
        normalized_bboxes[label] = [
            left_in / slide_width_in,
            top_in  / slide_height_in,
            width_in  / slide_width_in,
            height_in / slide_height_in,
        ]
    return normalized_bboxes

