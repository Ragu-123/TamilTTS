"""
Tamil Grapheme-to-Grapheme (G2G) Tokenizer
Maps continuous Tamil text to valid grapheme tokens matching IndicMFA dictionary.
"""
import os

PUNCT_SET = {' ', '\t', '\n', ',', '.', '-', '!', '?', ';', ':', '"', "'"}

def load_g2g_dictionary(dict_path):
    with open(dict_path, "r", encoding="utf-8") as f:
        return set([line.strip().split()[0] for line in f if line.strip()])

def segment_tamil_g2g(text, valid_set):
    tokens = []
    i = 0
    while i < len(text):
        if text[i] in PUNCT_SET:
            tokens.append(" ")
            i += 1
            continue
        matched = False
        for l in [3, 2, 1]:
            cand = text[i:i+l]
            if cand in valid_set:
                tokens.append(cand)
                i += l
                matched = True
                break
        if not matched:
            tokens.append(text[i])
            i += 1
    return " ".join([t for t in tokens if t.strip()])
