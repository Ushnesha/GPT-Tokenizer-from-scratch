# feeding strings into LLM
'''
In python, strings are unicode code points containing ~150K characters across 161 scripts (including emojis).
LLMs, however, only understand tokens, which are discrete units of text. To convert strings into tokens, 
we need to build a tokenizer from scratch.
'''

txt = "안녕하세요 👋 (hello in Korean!)"
print("="*100)
print([f"ord('{x}') = {ord(x)}" for x in txt])  # ord() gives the unicode code point of a character
print("="*100)
print(f"utf-8 raw bytes: {list(txt.encode('utf-8'))}")  # UTF-8 encoding of the string
print("="*100)
print(f"utf-16 raw bytes: {list(txt.encode('utf-16'))}")  # UTF-16 encoding of the string
print("="*100)
print(f"utf-32 raw bytes: {list(txt.encode('utf-32'))}")  # UTF-32 encoding of the string

'''
We can't use this unicode standards in LLMs because they have a limited vocabulary of tokens and 150K is huge number. Also, as we can see UTF-16/UTF-32
has a lot of 0s which are wasteful codes for an effcient LLM tokenizer. With UTF-8, there are only 256 possible byte values (vocabulary size), which 
is too small. So, a single sentence with many words can be stretched into a very long squence of tokens. With a finite small context size, this can 
lead to truncation of the input text, which is undesirable. Instead, we want to create a larger vocabulary size which we can tune as hyperparameter but
we want to stick to the UTF-8 encoding. eg. BPE
'''

'''
BPE: we find a pair of tokens in a sequence that occured most frequently. Once such pair is found, we create a new token for that pair, add it to the vocabulary and replace it in the sequence.
We continue such process untill we have the desired vocabulary size.
"aaabdaaabac" -> "ZabdaZabac" -> "ZYdZYdac" -> "XdXdac"
For example: starting sequence seq = "aaabdaaabac", vocabulary = {a, b, c, d}
1. find most frequent pair: "aa" (occurs 3 times)
2. create new token "Z" for "aa" and add to vocabulary: vocab = {a, b, c, d, Z}
3. replace all occurrences of "aa" with "Z" in the sequence: seq = "ZabdaZabac"
4. find most frequent pair: "ab" (occurs 2 times)
5. create new token "Y" for "ab" and add to vocabulary: vocab = {a, b, c, d, Z, Y}
6. replace all occurrences of "ab" with "Y" in the sequence: seq = "ZYdZYdac"
7. find most frequent pair: "ZY" (occurs 2 times)
8. create new token "X" for "ZY" and add to vocabulary: vocab = {a, b, c, d, Z, Y, X}
9. replace all occurrences of "ZY" with "X" in the sequence: seq = "XdXdac"
The sequence "XdXdac" cannot be compressed further by BPE - no pairs occur more than once.
To decompress the sequence, siple perform the reverse operation of BPE.
'''

# Implementation


class BytePairEncodingTokenizer:
    """
    A tokenizer implementing Byte Pair Encoding (BPE) algorithm.

    BPE iteratively finds the most frequent pair of tokens and merges them
    into a new token, building up a vocabulary from a base of UTF-8 bytes.
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size to build up to (default: 1000)
        """
        # Instance variables - vocabulary and mapping
        self.target_vocab_size = vocab_size
        self.vocab = set(i for i in range(256))  # initial vocab from UTF-8 bytes (0-255)
        self.current_vocab_size = len(self.vocab)
        self.decode_map = {}  # token -> (component1, component2) or token

    def _get_pairs(self, tokens: list) -> dict:
        """
        Find all adjacent token pairs and their frequencies.

        Args:
            tokens: List of token integers

        Returns:
            Dictionary with pairs as keys and frequencies as values
        """
        pairs = {}
        for i in range(len(tokens) - 1):
            pairs[(tokens[i], tokens[i + 1])] = pairs.get((tokens[i], tokens[i + 1]), 0) + 1
        return pairs

    def _merge_pairs(self, tokens: list, pair: tuple, new_token: int) -> list:
        """
        Replace all occurrences of a pair with a new token.

        Args:
            tokens: List of token integers
            pair: Tuple of two tokens to merge
            new_token: New token id to use

        Returns:
            Modified tokens list
        """
        i = 0
        while i < len(tokens) - 1:
            if (tokens[i], tokens[i + 1]) == pair:
                tokens[i] = new_token
                del tokens[i + 1]
            else:
                i += 1
        return tokens

    def _decompose_token(self, token: int) -> list:
        """
        Recursively decompose a token into its base UTF-8 bytes.

        Args:
            token: Token to decompose

        Returns:
            List of base byte tokens
        """
        if token < 256:
            return [token]

        dec_toks = self.decode_map[token]
        result = []
        result.extend(self._decompose_token(dec_toks[0]))
        result.extend(self._decompose_token(dec_toks[1]))
        return result

    def encode(self, text: str) -> list:
        """
        Encode text using BPE algorithm.

        Args:
            text: Text string to encode

        Returns:
            List of token integers
        """
        # Local variables for encoding process
        tokens = text.encode("utf-8")
        tokens = list(map(int, tokens))

        # Build vocabulary through BPE
        while self.current_vocab_size < self.target_vocab_size:
            paired_tokens = sorted(
                self._get_pairs(tokens).items(),
                key=lambda x: x[1],
                reverse=True
            )

            if not paired_tokens:
                break

            pair, freq = paired_tokens[0]
            new_token = self.current_vocab_size

            # Update all data structures
            tokens = self._merge_pairs(tokens, pair, new_token)
            self.vocab.add(new_token)
            self.decode_map[new_token] = pair
            self.current_vocab_size += 1

        return tokens

    def decode(self, tokens: list) -> list:
        """
        Decode tokens back to UTF-8 bytes.

        Args:
            tokens: List of token integers

        Returns:
            List of byte integers
        """
        dec_tokens = []
        for tok in tokens:
            dec_tokens.extend(self._decompose_token(tok))
        return dec_tokens

    def decode_v2(self, tokens: list) -> list:
        vocab_bytes = {i: bytes([i]) for i in range(256)}
        for idx, (p0, p1) in self.decode_map.items():
            vocab_bytes[idx] = vocab_bytes[p0] + vocab_bytes[p1]

        tokens = b"".join(vocab_bytes[tok] for tok in tokens)
        text = tokens.decode('utf-8', errors='replace')
            
        return text
        

    def decode_to_text(self, tokens: list) -> str:
        """
        Decode tokens directly to text string.

        Args:
            tokens: List of token integers

        Returns:
            Decoded text string
        """
        byte_tokens = self.decode(tokens)
        return bytes(byte_tokens).decode('utf-8')


# Example usage
if __name__ == "__main__":
    # Test text with diverse unicode characters
    test_text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception."
    tokens = test_text.encode("utf-8")
    tokens = list(map(int, tokens))
    print(f"Original tokens: {tokens}")
    print(f"Length of original tokens: {len(tokens)}")
    vocab_size = 276

    # Initialize tokenizer
    tokenizer = BytePairEncodingTokenizer(vocab_size=vocab_size)

    # Encode text
    encoded_tokens = tokenizer.encode(test_text)
    print("="*100)
    print(f"Encoded tokens: {encoded_tokens}")
    print(f"Length of encoded tokens with vocab size {vocab_size}: {len(encoded_tokens)}")

    # # Decode tokens
    # decoded_tokens = tokenizer.decode(encoded_tokens)
    # print("="*100)
    # print(f"Decoded tokens: {decoded_tokens}")
    # print(f"Length of decoded tokens: {len(decoded_tokens)}")

    # # Convert back to text
    # decoded_text = tokenizer.decode_to_text(encoded_tokens)
    # print("="*100)
    # print(f"Decoded text: {decoded_text}")
    # print("="*100)
    # print(f"Decoded text == original text: {decoded_text == test_text}")

    # Convert back to text
    decoded_text = tokenizer.decode_v2(encoded_tokens)
    print("="*100)
    print(f"Decoded text: {decoded_text}")
    print("="*100)
    print(f"Decoded text == original text: {decoded_text == test_text}")