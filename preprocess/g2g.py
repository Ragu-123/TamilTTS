"""
Tamil Grapheme-to-Grapheme (G2G) Tokenizer
Maps continuous Tamil text to valid grapheme tokens matching IndicMFA dictionary.
Token list is byte-compatible with the IndicMFA Tamil dictionary and vocab_size=384.
"""

PUNCT_SET = {' ', '\t', '\n', ',', '.', '-', '!', '?', ';', ':', '"', "'"}

# Base 270 G2G Tamil Aksharas & Tokens from IndicMFA Tamil Dictionary
TAMIL_G2G_TOKENS = [
    '<pad>', 'sil', '<unk>', ' ', 'spn', 'ஃ',
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ',
    'க', 'கா', 'கி', 'கீ', 'கு', 'கூ', 'கெ', 'கௌ', 'கே', 'கை', 'கொ', 'கோ', 'கௌ', 'க்', 'க்ஷ',
    'ங', 'ஙா', 'ஙி', 'ஙீ', 'ஙு', 'ஙூ', 'ஙெ', 'ஙே', 'ஙை', 'ஙொ', 'ஙோ', 'ஙௌ', 'ங்',
    'ச', 'சா', 'சி', 'சீ', 'சு', 'சூ', 'செ', 'சே', 'சை', 'சொ', 'சோ', 'சௌ', 'ச்',
    'ஞ', 'ஞா', 'ஞி', 'ஞீ', 'ஞு', 'ஞூ', 'ஞெ', 'ஞே', 'ஞை', 'ஞொ', 'ஞோ', 'ஞௌ', 'ஞ்',
    'ட', 'டா', 'டி', 'டீ', 'டு', 'டூ', 'டெ', 'டே', 'டை', 'டொ', 'டோ', 'டௌ', 'ட்',
    'ண', 'ணா', 'ணி', 'ணீ', 'ணு', 'ணூ', 'ணெ', 'ணே', 'ணை', 'ணொ', 'ணோ', 'ணௌ', 'ண்',
    'த', 'தா', 'தி', 'தீ', 'து', 'தூ', 'தெ', 'தே', 'தை', 'தொ', 'தோ', 'தௌ', 'த்',
    'ந', 'நா', 'நி', 'நீ', 'நு', 'நூ', 'நெ', 'நே', 'நை', 'நொ', 'நோ', 'நௌ', 'ந்',
    'ப', 'பா', 'பி', 'பீ', 'பு', 'பூ', 'பெ', 'பே', 'பை', 'பொ', 'போ', 'பௌ', 'ப்',
    'ம', 'மா', 'மி', 'மீ', 'மு', 'மூ', 'மெ', 'மே', 'மை', 'மொ', 'மோ', 'மௌ', 'ம்',
    'ய', 'யா', 'யி', 'யீ', 'யு', 'யூ', 'யெ', 'யே', 'யை', 'யொ', 'யோ', 'யௌ', 'ய்',
    'ர', 'ரா', 'ரி', 'ரீ', 'ரு', 'ரூ', 'ரெ', 'ரே', 'ரை', 'ரொ', 'ரோ', 'ரௌ', 'ர்',
    'ல', 'லா', 'லி', 'லீ', 'லு', 'லூ', 'லெ', 'லே', 'லை', 'லொ', 'லோ', 'லௌ', 'ல்',
    'வ', 'வா', 'வி', 'வீ', 'வு', 'வூ', 'வெ', 'வே', 'வை', 'வொ', 'வோ', 'வௌ', 'வ்',
    'ழ', 'ழா', 'ழி', 'ழீ', 'ழு', 'ழூ', 'ழெ', 'ழே', 'ழை', 'ழொ', 'ழோ', 'ழௌ', 'ழ்',
    'ள', 'ளா', 'ளி', 'ளீ', 'ளு', 'ளூ', 'ளெ', 'ளே', 'ளை', 'ளொ', 'ளோ', 'ளௌ', 'ள்',
    'ற', 'றா', 'றி', 'றீ', 'று', 'றூ', 'றெ', 'றே', 'றை', 'றொ', 'றோ', 'றௌ', 'ற்',
    'ன', 'னா', 'னி', 'னீ', 'னு', 'னூ', 'னெ', 'னே', 'னை', 'னொ', 'னோ', 'னௌ', 'ன்',
    'ஜ', 'ஜா', 'ஜி', 'ஜீ', 'ஜு', 'ஜூ', 'ஜெ', 'ஜே', 'ஜை', 'ஜொ', 'ஜோ', 'ஜௌ', 'ஜ்',
    'ஷ', 'ஷா', 'ஷி', 'ஷீ', 'ஷு', 'ஷூ', 'ஷெ', 'ஷே', 'ஷை', 'ஷொ', 'ஷோ', 'ஷௌ', 'ஷ்',
    'ஸ', 'ஸா', 'ஸி', 'ஸீ', 'ஸு', 'ஸூ', 'ஸெ', 'ஸே', 'ஸை', 'ஸொ', 'ஸோ', 'ஸௌ', 'ஸ்',
    'ஹ', 'ஹா', 'ஹி', 'ஹீ', 'ஹு', 'ஹூ', 'ஹெ', 'ஹே', 'ஹை', 'ஹொ', 'ஹோ', 'ஹௌ', 'ஹ்',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '.', ',', '!', '?', ';', ':', '-', "'", '"', '(', ')',
    # Standalone vowel signs observed in the MFA durations file (decomposed
    # aksharas from ASR-transcript mismatches). Kept as first-class tokens so the
    # dataset no longer maps them to <unk>.
    'ா', 'ி', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ', 'ௗ',
]

VOCAB_SIZE = 384


def load_g2g_dictionary(dict_path):
    """Load a whitespace-delimited MFA dictionary file into a token set.

    Args:
        dict_path: Path to IndicMFA-style lexicon file.
    Returns:
        Set[str]: Valid G2G tokens.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        return set(line.strip().split()[0] for line in f if line.strip())


def segment_tamil_g2g(text, valid_set):
    """Segment continuous Tamil text into space-separated G2G akshara tokens.

    Args:
        text (str): Raw Tamil text.
        valid_set (Set[str]): Valid G2G tokens (typically set(TAMIL_G2G_TOKENS)).
    Returns:
        str: Space-separated token string (no sil tokens; whitespace-separated words).
    """
    tokens = []
    i = 0
    while i < len(text):
        if text[i] in PUNCT_SET:
            tokens.append(" ")
            i += 1
            continue
        matched = False
        for l in [3, 2, 1]:
            cand = text[i:i + l]
            if cand in valid_set:
                tokens.append(cand)
                i += l
                matched = True
                break
        if not matched:
            tokens.append(text[i])
            i += 1
    return " ".join([t for t in tokens if t.strip()])


# Pauses that MFA transcribes with explicit 'sil' entries. Sentence-enders and
# clause commas both get an inserted 'sil' (MFA annotates phrase-final pauses of
# varying length with the same symbol; the duration head learns the length from
# data). Word-internal spaces are NOT marked — the durations file contains zero
# space tokens.
PAUSE_PUNCT = {'.', ',', '!', '?', ';', ':'}


def tokens_with_sil(text, valid_set):
    """Tokenize text into an ordered token list including 'sil' boundary markers.

    THE canonical inference-side tokenizer. Matches the MFA training convention:
    every utterance starts/ends with 'sil' and pause punctuation inserts 'sil'
    between clauses. Inference sequences therefore come from the same distribution
    the duration/pitch heads and decoder were trained on.

    Args:
        text (str): Raw Tamil text.
        valid_set (Set[str]): Valid G2G tokens.
    Returns:
        List[str]: Token sequence beginning and ending with 'sil'.
    """
    out = ["sil"]
    pending_pause = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in PAUSE_PUNCT:
            pending_pause = True
            i += 1
            continue
        if ch in PUNCT_SET:  # whitespace / quotes — no sil, just skipped
            i += 1
            continue
        if pending_pause and len(out) > 1:
            out.append("sil")
            pending_pause = False
        matched = False
        for l in [3, 2, 1]:
            cand = text[i:i + l]
            if cand in valid_set:
                out.append(cand)
                i += l
                matched = True
                break
        if not matched:
            out.append(ch)
            i += 1
    if len(out) > 1:
        out.append("sil")
    return out
