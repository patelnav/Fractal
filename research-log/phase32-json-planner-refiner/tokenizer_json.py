"""
JSON Tokenizer for the JSON Repair Engine.

Tokenizes JSON into structural tokens:
- Structural: { } [ ] : ,
- Literals: strings, numbers, booleans, null
- Special: <PAD> <MASK> <BOS> <EOS>

Key design decisions:
1. Strings are tokenized as single tokens (content preserved separately)
2. Numbers are tokenized as single tokens
3. Whitespace is stripped (not tokenized)
4. For training: use BPE-style subword tokenization for string contents
   For now: character-level inside strings with length limit
"""

import re
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class JSONToken:
    """A single JSON token with position info."""
    type: str       # 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'COLON', 'COMMA',
                    # 'STRING', 'NUMBER', 'TRUE', 'FALSE', 'NULL', 'SPECIAL'
    value: str      # The actual text (for strings/numbers) or symbol
    start: int      # Start position in original text
    end: int        # End position in original text


class JSONTokenizer:
    """
    Tokenizer for JSON with vocabulary management.

    Vocabulary:
    - Special tokens: <PAD>, <MASK>, <BOS>, <EOS>
    - Structural tokens: { } [ ] : ,
    - Literal tokens: true, false, null
    - String tokens: <STR_START> <STR_END> + character tokens
    - Number tokens: <NUM_START> <NUM_END> + digit/sign tokens

    For simplicity in v1: strings and numbers are atomic tokens with
    a small vocab of common values + <STRING>, <NUMBER> fallbacks.
    """

    # Core structural tokens
    STRUCTURAL = ['{', '}', '[', ']', ':', ',']

    # Literal tokens (JSON standard)
    LITERALS = ['true', 'false', 'null']

    # Python literal tokens (non-standard, need to learn to fix)
    PYTHON_LITERALS = ['True', 'False', 'None']

    # LLM artifact tokens
    LLM_ARTIFACTS = ['<FENCE>', '<FENCE_JSON>', '<COMMENT>', '<PROSE>']

    # Special tokens
    SPECIAL = ['<PAD>', '<MASK>', '<BOS>', '<EOS>']

    # Character tokens for strings (ASCII printable + common escapes)
    # Keep small for initial experiments
    CHAR_TOKENS = list(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~')

    # String wrapper tokens
    STRING_TOKENS = ['<STR>', '<STR_EMPTY>']  # <STR> wraps string content

    # Number tokens (digits + sign + decimal + exp)
    NUMBER_CHARS = list('0123456789+-.eE')
    NUMBER_TOKENS = ['<NUM>']  # <NUM> wraps number content

    def __init__(self, max_string_len: int = 32, max_number_len: int = 16):
        """
        Args:
            max_string_len: Maximum characters to keep from strings
            max_number_len: Maximum characters to keep from numbers
        """
        self.max_string_len = max_string_len
        self.max_number_len = max_number_len

        # Build vocabulary
        self.vocab = []

        # Special tokens first (fixed positions)
        self.vocab.extend(self.SPECIAL)

        # Structural tokens
        self.vocab.extend(self.STRUCTURAL)

        # Literals (JSON standard)
        self.vocab.extend(self.LITERALS)

        # Python literals (non-standard, to be fixed)
        self.vocab.extend(self.PYTHON_LITERALS)

        # LLM artifacts (to be stripped)
        self.vocab.extend(self.LLM_ARTIFACTS)

        # String tokens
        self.vocab.extend(self.STRING_TOKENS)

        # Number tokens
        self.vocab.extend(self.NUMBER_TOKENS)

        # Character tokens (for string contents)
        for c in self.CHAR_TOKENS:
            if c not in self.vocab:
                self.vocab.append(c)

        # Number character tokens (for number contents)
        for c in self.NUMBER_CHARS:
            if c not in self.vocab:
                self.vocab.append(c)

        # Build mappings
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        # Cache special token IDs
        self.pad_id = self.stoi['<PAD>']
        self.mask_id = self.stoi['<MASK>']
        self.bos_id = self.stoi['<BOS>']
        self.eos_id = self.stoi['<EOS>']
        self.str_id = self.stoi['<STR>']
        self.str_empty_id = self.stoi['<STR_EMPTY>']
        self.num_id = self.stoi['<NUM>']

        # Structural token IDs
        self.lbrace_id = self.stoi['{']
        self.rbrace_id = self.stoi['}']
        self.lbracket_id = self.stoi['[']
        self.rbracket_id = self.stoi[']']
        self.colon_id = self.stoi[':']
        self.comma_id = self.stoi[',']

        # Python literal IDs (non-standard)
        self.true_py_id = self.stoi['True']
        self.false_py_id = self.stoi['False']
        self.none_py_id = self.stoi['None']

        # LLM artifact IDs
        self.fence_id = self.stoi['<FENCE>']
        self.fence_json_id = self.stoi['<FENCE_JSON>']
        self.comment_id = self.stoi['<COMMENT>']
        self.prose_id = self.stoi['<PROSE>']

        self.special_ids = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}
        self.structural_ids = {self.lbrace_id, self.rbrace_id, self.lbracket_id,
                               self.rbracket_id, self.colon_id, self.comma_id}
        self.python_literal_ids = {self.true_py_id, self.false_py_id, self.none_py_id}
        self.artifact_ids = {self.fence_id, self.fence_json_id, self.comment_id, self.prose_id}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _lex_json(self, text: str) -> List[JSONToken]:
        """
        Lexically tokenize JSON text into JSONToken objects.

        Returns list of tokens with type, value, and position info.
        Handles LLM prose by skipping text before the first { or [ and
        after the last } or ].
        """
        # Find JSON boundaries (skip prose before/after)
        json_start = -1
        json_end = len(text)
        for idx, c in enumerate(text):
            if c in '{[':
                json_start = idx
                break
        for idx in range(len(text) - 1, -1, -1):
            if text[idx] in '}]':
                json_end = idx + 1
                break

        # If we found valid boundaries, trim the text
        if json_start > 0:
            text = text[json_start:json_end]

        tokens = []
        i = 0
        n = len(text)

        while i < n:
            c = text[i]

            # Skip whitespace
            if c in ' \t\n\r':
                i += 1
                continue

            # Structural tokens
            if c in '{}[]:,':
                type_map = {
                    '{': 'LBRACE', '}': 'RBRACE',
                    '[': 'LBRACKET', ']': 'RBRACKET',
                    ':': 'COLON', ',': 'COMMA'
                }
                tokens.append(JSONToken(type_map[c], c, i, i + 1))
                i += 1
                continue

            # String
            if c == '"':
                start = i
                i += 1
                chars = []
                while i < n and text[i] != '"':
                    if text[i] == '\\' and i + 1 < n:
                        # Handle escape sequences
                        esc = text[i + 1]
                        if esc == 'n':
                            chars.append('\n')
                        elif esc == 't':
                            chars.append('\t')
                        elif esc == 'r':
                            chars.append('\r')
                        elif esc == '\\':
                            chars.append('\\')
                        elif esc == '"':
                            chars.append('"')
                        elif esc == '/':
                            chars.append('/')
                        elif esc == 'b':
                            chars.append('\b')
                        elif esc == 'f':
                            chars.append('\f')
                        elif esc == 'u' and i + 5 < n:
                            # Unicode escape \uXXXX
                            try:
                                code = int(text[i + 2:i + 6], 16)
                                chars.append(chr(code))
                                i += 4
                            except ValueError:
                                chars.append(text[i:i + 2])
                        else:
                            chars.append(text[i:i + 2])
                        i += 2
                    else:
                        chars.append(text[i])
                        i += 1

                if i >= n:
                    # Unterminated string - include what we have
                    tokens.append(JSONToken('STRING', ''.join(chars), start, i))
                else:
                    i += 1  # Skip closing quote
                    tokens.append(JSONToken('STRING', ''.join(chars), start, i))
                continue

            # Number
            if c in '-0123456789':
                start = i
                # Match: -?[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?
                if c == '-':
                    i += 1
                # Integer part
                while i < n and text[i] in '0123456789':
                    i += 1
                # Decimal part
                if i < n and text[i] == '.':
                    i += 1
                    while i < n and text[i] in '0123456789':
                        i += 1
                # Exponent part
                if i < n and text[i] in 'eE':
                    i += 1
                    if i < n and text[i] in '+-':
                        i += 1
                    while i < n and text[i] in '0123456789':
                        i += 1
                tokens.append(JSONToken('NUMBER', text[start:i], start, i))
                continue

            # true
            if text[i:i + 4] == 'true':
                tokens.append(JSONToken('TRUE', 'true', i, i + 4))
                i += 4
                continue

            # false
            if text[i:i + 5] == 'false':
                tokens.append(JSONToken('FALSE', 'false', i, i + 5))
                i += 5
                continue

            # null
            if text[i:i + 4] == 'null':
                tokens.append(JSONToken('NULL', 'null', i, i + 4))
                i += 4
                continue

            # Python True (non-standard)
            if text[i:i + 4] == 'True':
                tokens.append(JSONToken('TRUE_PY', 'True', i, i + 4))
                i += 4
                continue

            # Python False (non-standard)
            if text[i:i + 5] == 'False':
                tokens.append(JSONToken('FALSE_PY', 'False', i, i + 5))
                i += 5
                continue

            # Python None (non-standard)
            if text[i:i + 4] == 'None':
                tokens.append(JSONToken('NONE_PY', 'None', i, i + 4))
                i += 4
                continue

            # Markdown fence with json (```json)
            if text[i:i + 7] == '```json':
                tokens.append(JSONToken('FENCE_JSON', '```json', i, i + 7))
                i += 7
                continue

            # Markdown fence (```)
            if text[i:i + 3] == '```':
                tokens.append(JSONToken('FENCE', '```', i, i + 3))
                i += 3
                continue

            # Line comment (//)
            if text[i:i + 2] == '//':
                start = i
                i += 2
                while i < n and text[i] != '\n':
                    i += 1
                tokens.append(JSONToken('COMMENT', text[start:i], start, i))
                continue

            # Block comment (/* ... */)
            if text[i:i + 2] == '/*':
                start = i
                i += 2
                while i < n - 1 and text[i:i + 2] != '*/':
                    i += 1
                if i < n - 1:
                    i += 2  # Skip closing */
                tokens.append(JSONToken('COMMENT', text[start:i], start, i))
                continue

            # Single-quoted string (non-standard)
            if c == "'":
                start = i
                i += 1
                chars = []
                while i < n and text[i] != "'":
                    if text[i] == '\\' and i + 1 < n:
                        # Handle escape sequences
                        esc = text[i + 1]
                        if esc == 'n':
                            chars.append('\n')
                        elif esc == 't':
                            chars.append('\t')
                        elif esc == "'":
                            chars.append("'")
                        elif esc == '\\':
                            chars.append('\\')
                        else:
                            chars.append(text[i:i + 2])
                        i += 2
                    else:
                        chars.append(text[i])
                        i += 1
                if i < n:
                    i += 1  # Skip closing quote
                tokens.append(JSONToken('STRING', ''.join(chars), start, i))
                continue

            # Unquoted identifier (non-standard key like {name: "value"})
            if c.isalpha() or c == '_':
                start = i
                while i < n and (text[i].isalnum() or text[i] == '_'):
                    i += 1
                ident = text[start:i]
                # Check if it's a known literal we might have missed
                if ident not in ('true', 'false', 'null', 'True', 'False', 'None'):
                    tokens.append(JSONToken('IDENT', ident, start, i))
                continue

            # Unknown character - skip but record position for error localization
            # In a real scenario, this might be the corruption we need to fix
            tokens.append(JSONToken('ERROR', c, i, i + 1))
            i += 1

        return tokens

    def tokenize(self, text: str, add_special: bool = True) -> List[int]:
        """
        Tokenize JSON text into token IDs.

        For strings: <STR> + character tokens (truncated to max_string_len)
        For numbers: <NUM> + digit tokens (truncated to max_number_len)

        Args:
            text: JSON text to tokenize
            add_special: If True, add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        lex_tokens = self._lex_json(text)

        ids = []
        if add_special:
            ids.append(self.bos_id)

        for tok in lex_tokens:
            if tok.type == 'LBRACE':
                ids.append(self.lbrace_id)
            elif tok.type == 'RBRACE':
                ids.append(self.rbrace_id)
            elif tok.type == 'LBRACKET':
                ids.append(self.lbracket_id)
            elif tok.type == 'RBRACKET':
                ids.append(self.rbracket_id)
            elif tok.type == 'COLON':
                ids.append(self.colon_id)
            elif tok.type == 'COMMA':
                ids.append(self.comma_id)
            elif tok.type == 'TRUE':
                ids.append(self.stoi['true'])
            elif tok.type == 'FALSE':
                ids.append(self.stoi['false'])
            elif tok.type == 'NULL':
                ids.append(self.stoi['null'])
            elif tok.type == 'TRUE_PY':
                ids.append(self.true_py_id)
            elif tok.type == 'FALSE_PY':
                ids.append(self.false_py_id)
            elif tok.type == 'NONE_PY':
                ids.append(self.none_py_id)
            elif tok.type == 'FENCE':
                ids.append(self.fence_id)
            elif tok.type == 'FENCE_JSON':
                ids.append(self.fence_json_id)
            elif tok.type == 'COMMENT':
                ids.append(self.comment_id)
            elif tok.type == 'IDENT':
                # Unquoted identifier - treat as a string
                if len(tok.value) == 0:
                    ids.append(self.str_empty_id)
                else:
                    ids.append(self.str_id)
                    for c in tok.value[:self.max_string_len]:
                        if c in self.stoi:
                            ids.append(self.stoi[c])
                    ids.append(self.str_id)
            elif tok.type == 'STRING':
                # Tokenize string content character by character
                if len(tok.value) == 0:
                    ids.append(self.str_empty_id)
                else:
                    ids.append(self.str_id)
                    for c in tok.value[:self.max_string_len]:
                        if c in self.stoi:
                            ids.append(self.stoi[c])
                        # else skip unknown chars
                    ids.append(self.str_id)  # Close string
            elif tok.type == 'NUMBER':
                # Tokenize number character by character
                ids.append(self.num_id)
                for c in tok.value[:self.max_number_len]:
                    if c in self.stoi:
                        ids.append(self.stoi[c])
                ids.append(self.num_id)  # Close number
            elif tok.type == 'ERROR':
                # Skip error tokens during tokenization
                pass

        if add_special:
            ids.append(self.eos_id)

        return ids

    def detokenize(self, ids: List[int]) -> str:
        """
        Convert token IDs back to JSON text.

        Args:
            ids: List of token IDs

        Returns:
            JSON text (may not be valid JSON if corrupted)
        """
        parts = []
        i = 0

        while i < len(ids):
            tok_id = ids[i]

            # Skip special tokens
            if tok_id in self.special_ids:
                i += 1
                continue

            tok = self.itos.get(tok_id, '')

            # Structural tokens
            if tok in self.STRUCTURAL:
                parts.append(tok)
                i += 1
                continue

            # Literals (JSON standard)
            if tok in self.LITERALS:
                parts.append(tok)
                i += 1
                continue

            # Python literals -> JSON literals (normalize)
            if tok == 'True':
                parts.append('true')
                i += 1
                continue
            if tok == 'False':
                parts.append('false')
                i += 1
                continue
            if tok == 'None':
                parts.append('null')
                i += 1
                continue

            # LLM artifacts - SKIP (strip completely)
            if tok_id in self.artifact_ids:
                i += 1
                continue

            # Empty string
            if tok_id == self.str_empty_id:
                parts.append('""')
                i += 1
                continue

            # String
            if tok_id == self.str_id:
                i += 1
                chars = []
                while i < len(ids) and ids[i] != self.str_id:
                    c = self.itos.get(ids[i], '')
                    # Skip special tokens and artifacts
                    if c and c not in ['<PAD>', '<MASK>', '<BOS>', '<EOS>', '<STR>', '<STR_EMPTY>', '<NUM>',
                                       '<FENCE>', '<FENCE_JSON>', '<COMMENT>', '<PROSE>',
                                       'True', 'False', 'None']:
                        # Escape special JSON characters
                        if c == '"':
                            chars.append('\\"')
                        elif c == '\\':
                            chars.append('\\\\')
                        elif c == '\n':
                            chars.append('\\n')
                        elif c == '\t':
                            chars.append('\\t')
                        elif c == '\r':
                            chars.append('\\r')
                        else:
                            chars.append(c)
                    i += 1
                if i < len(ids) and ids[i] == self.str_id:
                    i += 1  # Skip closing STR token
                parts.append('"' + ''.join(chars) + '"')
                continue

            # Number
            if tok_id == self.num_id:
                i += 1
                chars = []
                while i < len(ids) and ids[i] != self.num_id:
                    c = self.itos.get(ids[i], '')
                    if c in self.NUMBER_CHARS:
                        chars.append(c)
                    i += 1
                if i < len(ids) and ids[i] == self.num_id:
                    i += 1  # Skip closing NUM token
                parts.append(''.join(chars))
                continue

            # Unknown - skip
            i += 1

        # Add minimal spacing for readability
        result = []
        for i, part in enumerate(parts):
            result.append(part)
            # Add space after : and ,
            if part in [':', ',']:
                result.append(' ')

        return ''.join(result)

    def decode(self, ids: List[int]) -> str:
        """Alias for detokenize."""
        return self.detokenize(ids)

    def get_error_region(self, text: str, error_pos: int, window: int = 5) -> Tuple[int, int]:
        """
        Given an error position from json.loads(), find the token window around it.

        Args:
            text: Original JSON text
            error_pos: Character position of error
            window: Number of tokens on each side to include

        Returns:
            (start_idx, end_idx) token indices defining the error window
        """
        lex_tokens = self._lex_json(text)

        # Find the token containing or nearest to error_pos
        error_token_idx = 0
        for i, tok in enumerate(lex_tokens):
            if tok.start <= error_pos < tok.end:
                error_token_idx = i
                break
            elif tok.start > error_pos:
                error_token_idx = max(0, i - 1)
                break
        else:
            error_token_idx = len(lex_tokens) - 1

        # Define window
        start_idx = max(0, error_token_idx - window)
        end_idx = min(len(lex_tokens), error_token_idx + window + 1)

        return start_idx, end_idx


class JSONTokenizerV2:
    """
    Simplified JSON tokenizer - treats structural tokens atomically.

    This version is closer to how we want to do repairs:
    - Each structural token ({ } [ ] : ,) is one token
    - Each string literal is one token (with index into string table)
    - Each number literal is one token (with index into number table)
    - true/false/null are single tokens

    This gives much shorter sequences for typical JSON.
    """

    def __init__(self, num_string_buckets: int = 1000, num_number_buckets: int = 100):
        """
        Args:
            num_string_buckets: Number of unique string slots
            num_number_buckets: Number of unique number slots
        """
        # Build vocabulary
        self.vocab = [
            '<PAD>', '<MASK>', '<BOS>', '<EOS>',
            '{', '}', '[', ']', ':', ',',
            'true', 'false', 'null',
        ]

        # Add string tokens: <S0>, <S1>, ..., <S999>
        for i in range(num_string_buckets):
            self.vocab.append(f'<S{i}>')

        # Add number tokens: <N0>, <N1>, ..., <N99>
        for i in range(num_number_buckets):
            self.vocab.append(f'<N{i}>')

        self.num_string_buckets = num_string_buckets
        self.num_number_buckets = num_number_buckets

        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}

        # Special token IDs
        self.pad_id = self.stoi['<PAD>']
        self.mask_id = self.stoi['<MASK>']
        self.bos_id = self.stoi['<BOS>']
        self.eos_id = self.stoi['<EOS>']

        self.special_ids = {self.pad_id, self.mask_id, self.bos_id, self.eos_id}

        # String/number token ID ranges
        self.string_start_id = self.stoi['<S0>']
        self.string_end_id = self.stoi[f'<S{num_string_buckets - 1}>']
        self.number_start_id = self.stoi['<N0>']
        self.number_end_id = self.stoi[f'<N{num_number_buckets - 1}>']

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _hash_string(self, s: str) -> int:
        """Hash a string to a bucket index."""
        return hash(s) % self.num_string_buckets

    def _hash_number(self, n: str) -> int:
        """Hash a number string to a bucket index."""
        return hash(n) % self.num_number_buckets

    def tokenize(self, text: str, add_special: bool = True) -> Tuple[List[int], Dict[int, str]]:
        """
        Tokenize JSON text.

        Args:
            text: JSON text
            add_special: Add BOS/EOS

        Returns:
            (token_ids, value_table) where value_table maps position to actual string/number value
        """
        # Use the base tokenizer's lexer
        base = JSONTokenizer()
        lex_tokens = base._lex_json(text)

        ids = []
        value_table = {}  # Maps token position to actual value

        if add_special:
            ids.append(self.bos_id)

        for tok in lex_tokens:
            if tok.type == 'LBRACE':
                ids.append(self.stoi['{'])
            elif tok.type == 'RBRACE':
                ids.append(self.stoi['}'])
            elif tok.type == 'LBRACKET':
                ids.append(self.stoi['['])
            elif tok.type == 'RBRACKET':
                ids.append(self.stoi[']'])
            elif tok.type == 'COLON':
                ids.append(self.stoi[':'])
            elif tok.type == 'COMMA':
                ids.append(self.stoi[','])
            elif tok.type == 'TRUE':
                ids.append(self.stoi['true'])
            elif tok.type == 'FALSE':
                ids.append(self.stoi['false'])
            elif tok.type == 'NULL':
                ids.append(self.stoi['null'])
            elif tok.type == 'STRING':
                bucket = self._hash_string(tok.value)
                tok_id = self.string_start_id + bucket
                pos = len(ids)
                ids.append(tok_id)
                value_table[pos] = tok.value
            elif tok.type == 'NUMBER':
                bucket = self._hash_number(tok.value)
                tok_id = self.number_start_id + bucket
                pos = len(ids)
                ids.append(tok_id)
                value_table[pos] = tok.value

        if add_special:
            ids.append(self.eos_id)

        return ids, value_table

    def detokenize(self, ids: List[int], value_table: Optional[Dict[int, str]] = None) -> str:
        """
        Convert token IDs back to JSON.

        Args:
            ids: Token IDs
            value_table: Optional mapping of positions to actual values

        Returns:
            JSON text
        """
        parts = []

        for i, tok_id in enumerate(ids):
            if tok_id in self.special_ids:
                continue

            tok = self.itos.get(tok_id, '')

            if tok in ['{', '}', '[', ']', ':', ',']:
                parts.append(tok)
            elif tok in ['true', 'false', 'null']:
                parts.append(tok)
            elif tok.startswith('<S'):
                # String token
                if value_table and i in value_table:
                    val = value_table[i]
                    # Escape for JSON
                    escaped = val.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
                    parts.append(f'"{escaped}"')
                else:
                    parts.append('"<string>"')
            elif tok.startswith('<N'):
                # Number token
                if value_table and i in value_table:
                    parts.append(value_table[i])
                else:
                    parts.append('0')

        # Add spacing
        result = []
        for i, part in enumerate(parts):
            result.append(part)
            if part in [':', ',']:
                result.append(' ')

        return ''.join(result)


def test_tokenizer():
    """Test the JSON tokenizer."""
    tokenizer = JSONTokenizer()

    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.pad_id}, MASK={tokenizer.mask_id}, BOS={tokenizer.bos_id}, EOS={tokenizer.eos_id}")
    print()

    # Test cases
    test_jsons = [
        '{"name": "Alice", "age": 30}',
        '[1, 2, 3]',
        '{"nested": {"a": true, "b": null}}',
        '{"empty": "", "num": -3.14e10}',
        '{"broken": "unterminated',  # Intentionally broken
    ]

    for json_text in test_jsons:
        print(f"Original: {json_text}")
        ids = tokenizer.tokenize(json_text)
        print(f"Token IDs ({len(ids)}): {ids[:30]}{'...' if len(ids) > 30 else ''}")
        decoded = tokenizer.detokenize(ids)
        print(f"Decoded: {decoded}")
        print()

    # Test round-trip on valid JSON
    print("=== Round-trip test ===")
    valid_json = '{"users": [{"id": 1, "name": "Bob"}, {"id": 2, "name": "Carol"}], "count": 2}'
    ids = tokenizer.tokenize(valid_json)
    decoded = tokenizer.detokenize(ids)
    print(f"Original: {valid_json}")
    print(f"Decoded:  {decoded}")

    # Try parsing decoded
    try:
        parsed = json.loads(decoded)
        print(f"Parse OK: {parsed}")
    except json.JSONDecodeError as e:
        print(f"Parse error: {e}")


def test_tokenizer_v2():
    """Test the simplified tokenizer."""
    tokenizer = JSONTokenizerV2()

    print(f"\n=== V2 Tokenizer ===")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    json_text = '{"name": "Alice", "scores": [100, 95.5, 88]}'
    ids, value_table = tokenizer.tokenize(json_text)
    print(f"Original: {json_text}")
    print(f"Token IDs ({len(ids)}): {ids}")
    print(f"Value table: {value_table}")
    decoded = tokenizer.detokenize(ids, value_table)
    print(f"Decoded: {decoded}")


if __name__ == "__main__":
    test_tokenizer()
    test_tokenizer_v2()
