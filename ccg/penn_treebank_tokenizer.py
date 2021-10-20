import re

# this is inspired by the tokenizer from NLTK
# https://www.nltk.org/_modules/nltk/tokenize/treebank.html


STARTING_QUOTES = [
    (re.compile(r"^\""), r"``"),
    (re.compile(r"(``)"), r" \1 "),
    (re.compile(r"([ \(\[{<])(\"|\'{2})"), r"\1 `` "),
]

# punctuation
PUNCTUATION = [
    (re.compile(r"([:,])([^\d])"), r" \1 \2"),
    (re.compile(r"([:,])$"), r" \1 "),
    (re.compile(r"\.\.\."), r" ... "),
    (re.compile(r"[;@#$%&]"), r" \g<0> "),
    (
        re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'),
        r"\1 \2\3 ",
    ),  # Handles the final period.
    (re.compile(r"[?!]"), r" \g<0> "),
    (re.compile(r"([^'])' "), r"\1 ' "),
]

# Pads parentheses
PARENS_BRACKETS = (re.compile(r"[\]\[\(\)\{\}\<\>]"), r" \g<0> ")

# Optionally: Convert parentheses, brackets and converts them to PTB symbols.
CONVERT_PARENTHESES = [
    (re.compile(r"\("), "-LRB-"),
    (re.compile(r"\)"), "-RRB-"),
    (re.compile(r"\["), "-LSB-"),
    (re.compile(r"\]"), "-RSB-"),
    (re.compile(r"\{"), "-LCB-"),
    (re.compile(r"\}"), "-RCB-"),
]

DOUBLE_DASHES = (re.compile(r"--"), r" -- ")

# ending quotes
ENDING_QUOTES = [
    (re.compile(r'"'), " '' "),
    (re.compile(r"(\S)(\'\')"), r"\1 \2 "),
    (re.compile(r"([^' ])('[sS]|'[mM]|'[dD]|') "), r"\1 \2 "),
    (re.compile(r"([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1 \2 "),
]

# List of contractions adapted from Robert MacIntyre's tokenizer.
MacIntyreContractions_CONTRACTIONS2 = [
    r"(?i)\b(can)(?#X)(not)\b",
    r"(?i)\b(d)(?#X)('ye)\b",
    r"(?i)\b(gim)(?#X)(me)\b",
    r"(?i)\b(gon)(?#X)(na)\b",
    r"(?i)\b(got)(?#X)(ta)\b",
    r"(?i)\b(lem)(?#X)(me)\b",
    r"(?i)\b(more)(?#X)('n)\b",
    r"(?i)\b(wan)(?#X)(na)\s",
]
MacIntyreContractions_CONTRACTIONS3 = [r"(?i) ('t)(?#X)(is)\b", r"(?i) ('t)(?#X)(was)\b"]
CONTRACTIONS2 = list(map(re.compile, MacIntyreContractions_CONTRACTIONS2))
CONTRACTIONS3 = list(map(re.compile, MacIntyreContractions_CONTRACTIONS3))


def penn_tokenize(text, convert_parentheses=False):
    if type(text) == list:
        text = " ".join(text)
    for regexp, substitution in STARTING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp, substitution in PUNCTUATION:
        text = regexp.sub(substitution, text)

    # Handles parentheses.
    regexp, substitution = PARENS_BRACKETS
    text = regexp.sub(substitution, text)
    # Optionally convert parentheses
    if convert_parentheses:
        for regexp, substitution in CONVERT_PARENTHESES:
            text = regexp.sub(substitution, text)

    # Handles double dash.
    regexp, substitution = DOUBLE_DASHES
    text = regexp.sub(substitution, text)

    # add extra space to make things easier
    text = " " + text + " "

    for regexp, substitution in ENDING_QUOTES:
        text = regexp.sub(substitution, text)

    for regexp in CONTRACTIONS2:
        text = regexp.sub(r" \1 \2 ", text)
    for regexp in CONTRACTIONS3:
        text = regexp.sub(r" \1 \2 ", text)

    return text.split()