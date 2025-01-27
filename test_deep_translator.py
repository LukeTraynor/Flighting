try:
    from deep_translator import GoogleTranslator
    print("deep_translator is installed and available.")
except ModuleNotFoundError:
    print("deep_translator is not installed.")