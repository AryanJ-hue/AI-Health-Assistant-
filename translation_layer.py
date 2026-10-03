import argostranslate.package
import argostranslate.translate
from langdetect import detect, LangDetectException
_MODEL_READY = False
def ensure_hindi_model_installed():
    """Downloads and installs the Hindi->English package if not already
    present. Call this ONCE at app startup (or standalone before first
    use) — requires internet only for this one step."""
    global _MODEL_READY
    if _MODEL_READY:
        return
    installed_languages = argostranslate.translate.get_installed_languages()
    already_installed = any(
        lang.code == "hi" for lang in installed_languages
    )
    if already_installed:
        _MODEL_READY = True
        return
    print("Downloading Hindi->English translation model (one-time, needs internet)...")
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package = next(
        (p for p in available_packages if p.from_code == "hi" and p.to_code == "en"),
        None,
    )
    if package is None:
        raise RuntimeError(
            "Could not find the Hindi->English package in argos-translate's "
            "index. Check your internet connection and try again."
        )
    downloaded_path = package.download()
    argostranslate.package.install_from_path(downloaded_path)
    print("Hindi->English model installed. Future runs will be fully offline.")
    _MODEL_READY = True
class TranslationLayer:
    def __init__(self, target_lang: str = "en"):
        self.target_lang = target_lang

    def detect_language(self, text: str) -> str:
        """Returns an ISO 639-1 code (e.g. 'en', 'hi'), or 'unknown' if
        detection fails (very short input, mixed scripts, etc.)."""
        if any('\u0900' <= ch <= '\u097F' for ch in text):
            return "hi"
        detected = None
        try:
            detected = detect(text)
        except LangDetectException:
            return "unknown"
        HINGLISH_MARKERS = {"mujhe", "hai", "mera", "meri", "dard", "bukhar",
                             "khansi", "chakkar", "pet", "sir", "mein", "raha", "rahi"}
        words = set(text.lower().split())
        if detected in {"id", "tl", "so", "ms"} and words & HINGLISH_MARKERS:
            return "hi"
        return detected
    def process(self, text: str) -> dict:
        """
        Returns:
            original_text: what the user typed
            detected_language: best-guess language code
            translated_text: English version to feed into the extractor
                              (same as original_text if already English,
                              or if translation fails)
            translation_applied: bool, whether translation actually ran
        """
        detected = self.detect_language(text)
        if detected == "en":
            return {
                "original_text": text,
                "detected_language": "en",
                "translated_text": text,
                "translation_applied": False,
            }
        try:
            ensure_hindi_model_installed()
            translated = argostranslate.translate.translate(text, "hi", "en")
            return {
                "original_text": text,
                "detected_language": detected,
                "translated_text": translated,
                "translation_applied": True,
            }
        except Exception as e:
            # Fail-safe: if the model genuinely isn't installed/available,
            # fall through with the original text rather than crashing.
            print(f"WARNING: offline translation failed ({e}). Passing text through untranslated.")
            return {
                "original_text": text,
                "detected_language": detected,
                "translated_text": text,
                "translation_applied": False,
            }
if __name__ == "__main__":
    layer = TranslationLayer()
    test_cases = [
        "I have a sharp chest pain and shortness of breath.",
        "Mujhe tez bukhar aur khansi hai.",  # Hindi (Devanagari-transliterated/romanized)
        "मुझे सिर दर्द और चक्कर आ रहे हैं।",  # Hindi (Devanagari script)
    ]
    for text in test_cases:
        result = layer.process(text)
        print(f"\nOriginal ({result['detected_language']}): {result['original_text']}")
        print(f"Translated: {result['translated_text']}")
        print(f"Translation applied: {result['translation_applied']}")
