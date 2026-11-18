"""Deterministic text tools; no dataset fitting and no model thresholds here."""
import re
import unicodedata
from dataclasses import dataclass, asdict

PRODUCTS = ('ampul', 'süpürge', 'termostat', 'kamera', 'adaptör', 'sensör')
PRODUCT_CODES = re.compile(r'\b[A-Za-zÇĞİÖŞÜçğıöşü]{1,8}[- ]\d{2,8}(?:[-/][A-Za-z0-9]{1,6})?\b')

def normalize(text):
    # Keep casing, numbers and negation: BERTurk is a cased model.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()

def lower_tr(text):
    return text.replace('İ', 'i').replace('I', 'ı').lower()

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ca != cb)))
        previous = current
    return previous[-1]

def correct_products(text):
    changes = []
    def replace(match):
        word = match.group()
        lowered = lower_tr(word)
        if lowered in PRODUCTS or len(lowered) < 5:
            return word
        # Protect common inflected product forms: kameram, kameranın, sensörüm...
        suffixes = {'m','ım','im','um','üm','n','ın','in','un','ün','sı','si','su','sü',
                    'nın','nin','nun','nün','a','e','ya','ye','ı','i','u','ü',
                    'yı','yi','yu','yü','da','de','ta','te','dan','den','tan','ten',
                    'lar','ler','ları','leri','ımın','imin','umun','ümün','mın','min'}
        if any(lowered.startswith(p) and lowered[len(p):] in suffixes for p in PRODUCTS):
            return word
        candidates = [p for p in PRODUCTS if edit_distance(lowered, p) == 1]
        if len(candidates) != 1:
            return word
        replacement = candidates[0]
        changes.append({'original': word, 'suggested': replacement})
        return replacement
    # Preserve model codes even if they contain alphabetic pieces.
    pieces, last = [], 0
    for code in PRODUCT_CODES.finditer(text):
        pieces.append(re.sub(r'[^\W\d_]+', replace, text[last:code.start()], flags=re.UNICODE))
        pieces.append(code.group())
        last = code.end()
    pieces.append(re.sub(r'[^\W\d_]+', replace, text[last:], flags=re.UNICODE))
    return ''.join(pieces), changes

def small_talk(text):
    key = lower_tr(normalize(text)).strip(' .!?…,:;')
    greetings = {'merhaba','selam','selamlar','hello','hi','hey','günaydın','iyi akşamlar','iyi günler','kolay gelsin'}
    if key in greetings:
        return 'Merhaba! Akıllı ev cihazınızla ilgili destek talebinizi yazabilirsiniz.'
    if key in {'nasılsın','nasılsınız'}:
        return 'Teşekkür ederim, yardımcı olmaya hazırım. Cihazınızla ilgili sorununuz nedir?'
    if key in {'teşekkürler','teşekkür ederim','sağ ol','sağol'}:
        return 'Rica ederim. Başka bir destek talebiniz varsa yazabilirsiniz.'
    return None

def inspect_text(text):
    normalized = normalize(text)
    suggested, changes = correct_products(normalized)
    codes = list(dict.fromkeys(m.group().upper().replace(' ', '-') for m in PRODUCT_CODES.finditer(normalized)))
    return {'normalized': normalized, 'suggested': suggested, 'corrections': changes, 'model_codes': codes, 'rule_reply': small_talk(normalized)}
