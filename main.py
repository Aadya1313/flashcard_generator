import requests
from bs4 import BeautifulSoup
def fetch_wikipedia_intro(topic):
    """Fetch the introductory paragraphs from Wikipedia for a given topic."""
    url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"[ERROR] Could not fetch Wikipedia page for topic: {topic}")
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.select("p")
    text = ""
    for p in paragraphs:
        if len(p.text.strip()) > 50:
            text += p.text.strip() + " "
        if len(text.split()) > 300:  # limit to first ~300 words
            break
    return text
import os
import easyocr
import pytesseract
import cv2
from PIL import Image
from utils import create_flashcard

# For free GPT-like models
from transformers import pipeline

# Set up paths
INPUT_DIR = "images/"
OUTPUT_DIR = "images/output_flashcards/"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load free GPT-like model for QnA and summarization (distilGPT2 for demo, replace with better if available)
gpt_pipe = pipeline('text-generation', model='distilgpt2')

CANDIDATE_LABELS = ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'General', 'History', 'Computer Science', 'Political Science']

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
    temp_path = image_path + '_preprocessed.png'
    cv2.imwrite(temp_path, image)
    return temp_path

def extract_text(image_path):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image_path, detail=0)
    text = '\n'.join(result)
    if len(text.strip().split()) < 5:
        img = cv2.imread(image_path)
        text = pytesseract.image_to_string(img)
    return text

def smart_generate_flashcard_qa(text):
    # Use GPT-like model to generate Q&A pairs (demo: generate 3 Q&A)
    prompt = f"Extract 3 important question-answer flashcards from the following notes:\n{text}\nFormat: Q: ... A: ..."
    gpt_output = gpt_pipe(prompt, max_length=256, num_return_sequences=1)[0]['generated_text']
    # Parse Q&A pairs
    flashcards = []
    for line in gpt_output.split('\n'):
        if line.strip().startswith('Q:') and 'A:' in line:
            q, a = line.split('A:', 1)
            flashcards.append((q.replace('Q:', '').strip(), a.strip()))
    if not flashcards:
        # fallback to rule-based
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for line in lines:
            flashcards.append(("What is this note about?", line))
    return flashcards

def classify_subject_gpt(text):
    # Simple keyword-based subject classifier (fallback if not using zero-shot model)
    text_lower = text.lower()
    if any(word in text_lower for word in ['math', 'algebra', 'equation', 'theorem']):
        return 'Mathematics'
    elif any(word in text_lower for word in ['physics', 'force', 'energy', 'motion']):
        return 'Physics'
    elif any(word in text_lower for word in ['chemistry', 'reaction', 'atom', 'molecule']):
        return 'Chemistry'
    elif any(word in text_lower for word in ['biology', 'cell', 'organism', 'gene']):
        return 'Biology'
    else:
        return 'General'

def process_images():
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(INPUT_DIR, filename)
            print(f"Processing {filename}...")
            preprocessed_path = preprocess_image(filepath)
            text = extract_text(preprocessed_path)
            print("Extracted text:", text)
            subject = classify_subject_gpt(text)
            flashcards = smart_generate_flashcard_qa(text)
            for i, (q, a) in enumerate(flashcards):
                flashcard_path = os.path.join(OUTPUT_DIR, f"{subject}_flashcard_{i+1}_{filename}")
                create_flashcard(f"Q: {q}\nA: {a}", flashcard_path)
                print(f"Flashcard saved to {flashcard_path}")
            os.remove(preprocessed_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        topic = ' '.join(sys.argv[1:])
        print(f"[INFO] Fetching content for topic: {topic}")
        text = fetch_wikipedia_intro(topic)
        if not text.strip():
            print(f"[ERROR] No content found for topic: {topic}")
        else:
            # Remove Wikipedia reference markers like [1], [2][3], etc.
            import re
            cleaned_text = re.sub(r'\[\d+(?:\]\[\d+)*\]', '', text)
            cleaned_text = re.sub(r'\[\d+\]', '', cleaned_text)
            print("[INFO] Extracted text (first 500 chars):", cleaned_text[:500], "...")
            subject = classify_subject_gpt(cleaned_text)
            # Split text into 3-4 chunks for flashcards
            sentences = re.split(r'(?<=[.!?]) +', cleaned_text.strip())
            n = max(3, min(5, len(sentences)))
            chunk_size = max(1, len(sentences) // n)
            flashcard_chunks = [" ".join(sentences[i*chunk_size:(i+1)*chunk_size]).strip() for i in range(n)]
            for i, chunk in enumerate(flashcard_chunks):
                if chunk:
                    flashcard_path = os.path.join(OUTPUT_DIR, f"{subject}_flashcard_{i+1}_from_web.png")
                    create_flashcard(chunk, flashcard_path)
                    print(f"Flashcard saved to {flashcard_path}")
    else:
        process_images()