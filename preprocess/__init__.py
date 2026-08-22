"""
TamilTTS Preprocessing & Forced Alignment Pipeline
==================================================
Industry standard data alignment, G2G tokenization, and duration extraction
powered by AI4Bharat IndicMFA.
"""
from .g2g import segment_tamil_g2g, load_g2g_dictionary
from .align import run_mfa_alignment, extract_durations_from_textgrid
