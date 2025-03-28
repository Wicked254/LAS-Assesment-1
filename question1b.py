# !pip install transformers
from transformers import pipeline

# Initialize translation pipelines
en_to_sw = pipeline('translation', model='Helsinki-NLP/opus-mt-en-swc')
sw_to_en = pipeline('translation', model='Helsinki-NLP/opus-mt-swc-en')

def translate(text, target_language):
    """Translate text between English and Swahili."""
    if target_language == 'sw':
        result = en_to_sw(text)
    elif target_language == 'en':
        result = sw_to_en(text)
    else:
        return "Unsupported language"
    return result[0]['translation_text']

# Test translations
english_text = "Hello, how are you?"
swahili_text = "Habari yako?"

# English to Swahili
translated_sw = translate(english_text, 'sw')
print(f"English -> Swahili: {translated_sw}")

# Swahili to English
translated_en = translate(swahili_text, 'en')
print(f"Swahili -> English: {translated_en}")