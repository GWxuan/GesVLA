import logging
import re
from typing import Tuple

import numpy as np
import sentencepiece

import openpi.shared.download as download

# Special token IDs for the PaliGemma vocabulary.
END_OF_PREFIX_TOKEN = 257022
BEGIN_OF_ACTION = 257021
BEGIN_OF_REASONING = 257020
PALIGEMMA_EOS_TOKEN = 1


class DiscreteCoordinateProcessor:
    """Coordinate discretization processor.
    
    Maps continuous coordinate values to discrete token bins and back,
    enabling coordinate representation within the PaliGemma vocabulary.
    """
    
    def __init__(self, num_bins: int = 256, coord_range: Tuple[float, float] = (0, 1)):
        self.num_bins = num_bins
        self.coord_range = coord_range
        self.min_val, self.max_val = coord_range
        self.range_size = self.max_val - self.min_val
        
        # Use the least-frequent num_bins tokens from the vocabulary.
        self.COORD_BIN_START = 256000 - num_bins  # e.g. 255744
        
    def discretize_coordinate(self, coordinate: float) -> int:
        """Map a continuous coordinate value to a discrete bin index."""
        normalized = (coordinate - self.min_val) / self.range_size
        bin_index = int(normalized * (self.num_bins - 1))
        return max(0, min(self.num_bins - 1, bin_index))
    
    def undiscretize_coordinate(self, bin_index: int) -> float:
        """Map a discrete bin index back to a continuous coordinate value."""
        normalized = bin_index / (self.num_bins - 1)
        return self.min_val + normalized * self.range_size
    
    def bin_to_token(self, bin_index: int) -> int:
        """Convert a bin index to the corresponding token ID."""
        return self.COORD_BIN_START + bin_index
    
    def token_to_bin(self, token: int) -> int:
        """Convert a token ID to the corresponding bin index."""
        return token - self.COORD_BIN_START
    
    def is_coord_token(self, token: int) -> bool:
        """Check whether a token ID falls within the coordinate bin range."""
        return self.COORD_BIN_START <= token < (self.COORD_BIN_START + self.num_bins)

class FusePaligemmaTokenizer:
    def __init__(self, max_len: int = 48, coord_bins: int = 256, coord_range: Tuple[float, float] = (0, 1)):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        
        self._coord_processor = DiscreteCoordinateProcessor(
            num_bins=coord_bins, 
            coord_range=coord_range
        )
        
        self._validate_token_range()

    def _validate_token_range(self):
        """Verify that coordinate tokens fit within the vocabulary."""
        vocab_size = self._tokenizer.vocab_size()
        max_coord_token = self._coord_processor.COORD_BIN_START + self._coord_processor.num_bins - 1
        
        if max_coord_token >= vocab_size:
            raise ValueError(
                f"Coordinate token range exceeds vocabulary: max_coord_token={max_coord_token}, "
                f"vocab_size={vocab_size}"
            )

    def _segment_and_encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list:
        """Segment text by coordinate patterns and encode each piece.

        Plain text segments are encoded via sentencepiece; coordinate pairs
        are directly converted to discrete coordinate token IDs.

        Returns:
            A list of integer token IDs (mixed sentencepiece and coordinate tokens).
        """
        coord_pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
        pieces = []
        last_idx = 0
        matches = list(re.finditer(coord_pattern, text))
        
        # If there are no coordinates, encode the text as a single piece.
        if not matches:
            return self._tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
        
        first_piece = True
        for i, m in enumerate(matches):
            start, end = m.span()
            before = text[last_idx:start]
            if before != "" or first_piece:
                # Encode the 'before' segment; only the first non-empty segment gets BOS.
                bos_flag = add_bos and first_piece
                eos_flag = False  # Intermediate segments never get EOS.
                if before != "":
                    encoded_before = self._tokenizer.encode(before, add_bos=bos_flag, add_eos=eos_flag)
                else:
                    # Empty string but may need BOS.
                    encoded_before = self._tokenizer.encode("", add_bos=bos_flag, add_eos=eos_flag)
                pieces.extend(encoded_before)
                first_piece = False
            
            # Process coordinate pair directly (bypass sentencepiece).
            try:
                x, y = float(m.group(1)), float(m.group(2))
                x_bin = self._coord_processor.discretize_coordinate(x)
                y_bin = self._coord_processor.discretize_coordinate(y)
                x_token = self._coord_processor.bin_to_token(x_bin)
                y_token = self._coord_processor.bin_to_token(y_bin)
                pieces.append(x_token)
                pieces.append(y_token)
            except Exception as e:
                logging.warning(f"Coordinate parsing failed (fallback to plain text): {m.group(0)} error: {e}")
                # On failure, fall back to encoding the coordinate literal as plain text.
                encoded_literal = self._tokenizer.encode(m.group(0), add_bos=False, add_eos=False)
                pieces.extend(encoded_literal)
            
            last_idx = end
        
        # Process tail segment.
        tail = text[last_idx:]
        if tail != "":
            # Tail segment: add eos if this is the last segment and add_eos is set.
            encoded_tail = self._tokenizer.encode(tail, add_bos=False, add_eos=add_eos)
            pieces.extend(encoded_tail)
        else:
            # Empty tail, but may still need an EOS token.
            if add_eos:
                # force an EOS by encoding empty with add_eos
                pieces.extend(self._tokenizer.encode("", add_bos=False, add_eos=True))
        
        return pieces

    def tokenize(self,
                 thought: list[str],
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a thought pair into token arrays with associated masks."""
        prefix = thought[0]
        suffix = thought[1] if len(thought) > 1 else ""

        # Encode prefix (with BOS) and suffix segments using coordinate-aware encoding.
        prefix_tokens = self._segment_and_encode(prefix, add_bos=True, add_eos=False) + [END_OF_PREFIX_TOKEN]

        if len(thought) > 1:
            # reasoning: prefix + BEGIN_OF_REASONING + encoded(suffix with eos)
            suffix_encoded = self._segment_and_encode(suffix, add_bos=False, add_eos=True)
            suffix_tokens = [BEGIN_OF_REASONING] + suffix_encoded
            diffusion_loss_mask = np.False_
        else:
            # action-only: just BEGIN_OF_ACTION (no suffix text)
            suffix_tokens = [BEGIN_OF_ACTION]
            diffusion_loss_mask = np.True_

        tokens = prefix_tokens + suffix_tokens

        # Create masks.
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(suffix_tokens)

        # text_loss_mask: no loss on prefix, loss on all suffix tokens.
        text_loss_mask = [False] * len(prefix_tokens) + [True] * len(suffix_tokens)

        # Pad or truncate to max length.
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [0] * (self._max_len - tokens_len)
            padding_mask = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding_mask
            ar_mask = ar_mask + padding_mask
            text_loss_mask = text_loss_mask + padding_mask
        else:
            if len(tokens) > self._max_len:
                logging.warning(f"Token length exceeds max length, truncating.")
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            text_loss_mask = text_loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(text_loss_mask),
            diffusion_loss_mask,
        )
    
    def extract_thoughts(self, tokens: np.ndarray) -> str:
        """Extract thoughts from mixed token list (coord tokens are converted to continuous coordinates)."""
        tokens = tokens.tolist()
        filtered_tokens = []
        
        for t in tokens:
            if t == PALIGEMMA_EOS_TOKEN:
                break
            filtered_tokens.append(t)
        i = 0
        n = len(filtered_tokens)
        parts = []
        current_chunk = []
        
        while i < n:
            t = filtered_tokens[i]
            if self._coord_processor.is_coord_token(t):
                # flush current chunk
                if current_chunk:
                    parts.append(self._tokenizer.decode(current_chunk))
                    current_chunk = []
                # try to read pair (x_token, y_token)
                if i + 1 < n and self._coord_processor.is_coord_token(filtered_tokens[i + 1]):
                    x_token = filtered_tokens[i]
                    y_token = filtered_tokens[i + 1]
                    x_bin = self._coord_processor.token_to_bin(x_token)
                    y_bin = self._coord_processor.token_to_bin(y_token)
                    x_cont = self._coord_processor.undiscretize_coordinate(x_bin)
                    y_cont = self._coord_processor.undiscretize_coordinate(y_bin)
                    parts.append(f"({x_cont:.3f},{y_cont:.3f})")
                    i += 2
                else:
                    # Unpaired coordinate token — fall back to plain decode.
                    parts.append(self._tokenizer.decode([t]))
                    i += 1
            else:
                # Regular token: accumulate into chunk until a coord token or end.
                current_chunk.append(t)
                i += 1
        
        if current_chunk:
            parts.append(self._tokenizer.decode(current_chunk))
        
        # Concatenate all segments.
        return "".join(parts)


