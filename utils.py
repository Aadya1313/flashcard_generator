import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import textwrap
import math

def preprocess_image(image_path):
    """Preprocess the image for better OCR results."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def extract_text(image):
    """Extract text from the preprocessed image using Tesseract OCR."""
    return pytesseract.image_to_string(image)

def create_flashcard(text, output_path):
    """Create a flashcard image with up to 5 bullet points, each grouping several sentences/lines from the image."""
    width, height = 800, 400
    flashcard = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(flashcard)
    # Try to use a TTF font for better appearance
    font = ImageFont.load_default()
    margin = 40
    offset = 40
    line_spacing = 8
    max_width = width - 2 * margin

    # Extract non-empty lines (sentences) from the OCR result
    lines = [l.strip() for l in text.split('\n') if l.strip() and len(l.strip().split()) > 2]
    if not lines:
        lines = [text.strip()]
    n_bullets = min(5, len(lines))
    if n_bullets == 0:
        flashcard.save(output_path)
        return
    group_size = math.ceil(len(lines) / n_bullets)
    bullet_groups = [lines[i*group_size:(i+1)*group_size] for i in range(n_bullets)]

    for group in bullet_groups:
        bullet_text = ' '.join(group)
        bullet_sentence = '\u2022 ' + bullet_text
        wrapped = textwrap.wrap(bullet_sentence, width=60)
        for wrap_line in wrapped:
            # Center align: calculate text width and set x accordingly
            bbox = draw.textbbox((0, 0), wrap_line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (width - text_width) // 2
            draw.text((x, offset), wrap_line, fill='black', font=font)
            line_height = bbox[3] - bbox[1]
            offset += line_height + line_spacing
            if offset > height - margin:
                break
        if offset > height - margin:
            break
    flashcard.save(output_path)