import streamlit as st
import importlib.util
import sys
import pandas as pd
import time

# --- Setup Page Config ---
st.set_page_config(
    page_title="BPE Tokenizer Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Inject CSS for Rich & Premium Glassmorphism Aesthetics ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* Apply primary and secondary fonts */
html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stTextArea {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main gradient Title styling */
.main-title {
    background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    font-family: 'Outfit', sans-serif;
    text-align: center;
    letter-spacing: -0.02em;
}

.subtitle {
    text-align: center;
    color: #94A3B8;
    font-size: 1.1rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Glassmorphism Card panel */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 24px;
    transition: transform 0.3s ease, border-color 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    border-color: rgba(168, 85, 247, 0.3);
}

/* Metric Display CSS */
.metrics-row {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 10px;
    margin-bottom: 10px;
}

.metric-card {
    flex: 1;
    min-width: 180px;
    background: rgba(15, 23, 42, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    background: rgba(168, 85, 247, 0.05);
    border-color: rgba(168, 85, 247, 0.2);
}

.metric-val {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #A855F7, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Outfit', sans-serif;
    margin-bottom: 6px;
}

.metric-lbl {
    font-size: 0.8rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* Token visualization chip elements */
.token-chip {
    display: inline-block;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.4);
    border-radius: 8px;
    padding: 4px 10px;
    margin: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.95rem;
    color: #94A3B8; /* Readable dark purple text for light theme */
    transition: all 0.2s ease;
    
    /* Prevent overflow for long merged tokens */
    max-width: 100%;
    word-break: break-all;
    white-space: normal;
}

.token-chip:hover {
    background: rgba(236, 72, 153, 0.2);
    border-color: rgba(236, 72, 153, 0.6);
    transform: scale(1.05);
}

.token-chip-id {
    font-size: 0.75rem;
    color: #EC4899;
    margin-left: 5px;
    vertical-align: middle;
    font-weight: bold;
}

/* System theme adjustments for dark mode */
@media (prefers-color-scheme: dark) {
    .token-chip {
        color: #F8FAFC; /* Readable light text for dark theme */
    }
    .token-chip-id {
        color: #F472B6; /* Light pink ID for dark theme */
    }
}

/* Section labels */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
    color: var(--text-color, #94A3B8);
    border-left: 4px solid #A855F7;
    padding-left: 10px;
}
</style>
""", unsafe_allow_html=True)

# --- Load the Tokenizer Class Dynamically (handling hyphenated module name) ---
try:
    spec = importlib.util.spec_from_file_location("tokenizer_module", "Build-gpt2-tokenizer-from-scratch.py")
    tokenizer_module = importlib.util.module_from_spec(spec)
    sys.modules["tokenizer_module"] = tokenizer_module
    spec.loader.exec_module(tokenizer_module)
    BytePairEncodingTokenizer = tokenizer_module.BytePairEncodingTokenizer
except Exception as e:
    st.error(f"Failed to load tokenizer module: {e}")
    st.stop()


# --- Streamlit Layout ---
st.markdown('<div class="main-title">🧠 BPE Tokenizer Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An interactive visual sandbox to study Byte Pair Encoding (BPE) tokenization from scratch</div>', unsafe_allow_html=True)

# Initialize Sidebar Inputs
st.sidebar.markdown("### ⚙️ Tokenizer Settings")

vocab_size = st.sidebar.slider(
    "Target Vocabulary Size",
    min_value=256,
    max_value=2000,
    value=350,
    step=1,
    help="Size of the vocabulary to build up to. Starts at 256 for basic raw UTF-8 bytes."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 BPE Visual Reference")
st.sidebar.info(
    "Byte Pair Encoding (BPE) finds the most frequent pair of adjacent bytes/tokens "
    "in your text and merges them into a new token. This loop runs iteratively until the target "
    "vocabulary size is reached."
)

st.sidebar.markdown("### ⚠️ Performance Guidelines")
st.sidebar.warning(
    "Because BPE is implemented from scratch in pure Python without index structures, "
    "the execution complexity is high. For quick results, keep inputs under **1,500 characters** "
    "and vocab sizes under **800**."
)

# Main container columns
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.markdown('<div class="section-header">📝 Input Text</div>', unsafe_allow_html=True)
    
    # Pre-populate with a rich unicode sample
    sample_text = (
        "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 "
        "The very name strikes fear and awe into the hearts of programmers worldwide. "
        "We all know we ought to “support Unicode” in our software. "
        "But Unicode can be abstruse, and diving into the thousand-page Unicode Standard can be more than a little intimidating."
    )
    
    text_input = st.text_area(
        "Enter text to tokenize:",
        value=sample_text,
        height=240,
        placeholder="Type or paste text here..."
    )
    
    # Text length warnings
    char_len = len(text_input)
    byte_len = len(text_input.encode("utf-8"))
    
    st.markdown(f"**Current Input Statistics:** `{char_len}` characters | `{byte_len}` raw UTF-8 bytes")
    
    if char_len > 1500 or vocab_size > 1000:
        st.warning("⚠️ Warning: Large text or high vocab size might cause a short delay in calculation.")
        
    submit_btn = st.button("🚀 Run BPE Tokenization", use_container_width=True)

# Run Tokenizer Logic
if submit_btn or 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    if not text_input.strip():
        st.info("Please enter some text in the text area to start.")
    else:
        with st.spinner("Training BPE vocabulary and encoding text..."):
            start_time = time.time()
            
            # 1. Instantiate the tokenizer with the target vocab size
            tokenizer = BytePairEncodingTokenizer(vocab_size=vocab_size)
            
            # 2. Encode
            encoded_tokens = tokenizer.encode(text_input)
            
            # 3. Decode
            decoded_text = tokenizer.decode(encoded_tokens)
            decoded_bytes = list(b"".join(tokenizer.vocab_bytes[tok] for tok in encoded_tokens))
            
            elapsed_time = time.time() - start_time
            
            # Save results into session state to keep them persistent across sidebar interaction
            st.session_state['encoded_tokens'] = encoded_tokens
            st.session_state['decoded_bytes'] = decoded_bytes
            st.session_state['decoded_text'] = decoded_text
            st.session_state['elapsed_time'] = elapsed_time
            st.session_state['vocab_size'] = tokenizer.current_vocab_size
            st.session_state['vocab_bytes'] = tokenizer.vocab_bytes
            
            # Build vocabulary table
            learned_vocab = []
            for t in range(256, tokenizer.current_vocab_size):
                if t in tokenizer.decode_map:
                    pair = tokenizer.decode_map[t]
                    # Get the raw bytes for this token
                    token_bytes = tokenizer.vocab_bytes.get(t, b"")
                    char_repr = token_bytes.decode('utf-8', errors='replace')
                    
                    # Highlight control/space characters for clarity
                    char_repr_escaped = char_repr.replace(" ", "␣").replace("\n", "↵")
                    
                    learned_vocab.append({
                        "Token ID": t,
                        "Merged Pair": f"({pair[0]}, {pair[1]})",
                        "Decompressed Bytes": str(list(token_bytes)),
                        "Text Representation": char_repr_escaped
                    })
            st.session_state['learned_vocab'] = learned_vocab

# Display results if available in session state
if 'encoded_tokens' in st.session_state:
    encoded_tokens = st.session_state['encoded_tokens']
    decoded_bytes = st.session_state['decoded_bytes']
    decoded_text = st.session_state['decoded_text']
    elapsed_time = st.session_state['elapsed_time']
    actual_vocab_size = st.session_state['vocab_size']
    learned_vocab = st.session_state['learned_vocab']
    
    original_byte_len = len(text_input.encode("utf-8"))
    encoded_token_len = len(encoded_tokens)
    decoded_token_len = len(decoded_bytes)
    
    # Calculate compression ratio
    compression_ratio = original_byte_len / encoded_token_len if encoded_token_len > 0 else 1.0
    
    # --- Columns for Layout Output ---
    with col_right:
        st.markdown('<div class="section-header">📊 Summary Metrics</div>', unsafe_allow_html=True)
        
        # Injected Custom Cards for premium styling
        st.markdown(f"""
        <div class="metrics-row">
            <div class="metric-card">
                <div class="metric-val">{original_byte_len}</div>
                <div class="metric-lbl">Original Bytes</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{encoded_token_len}</div>
                <div class="metric-lbl">Encoded Tokens</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{decoded_token_len}</div>
                <div class="metric-lbl">Decoded Bytes</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{compression_ratio:.2f}x</div>
                <div class="metric-lbl">Compression Ratio</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"⏱️ **BPE learning & encoding completed in:** `{elapsed_time:.4f} seconds` | Learned vocabulary: `{actual_vocab_size}` items.")

    # Bottom Area: Detailed Interactive Tabs
    st.markdown("---")
    st.markdown('<div class="section-header">🔍 Deep Dive Analysis</div>', unsafe_allow_html=True)
    
    tab_encoded, tab_vocab, tab_decoded = st.tabs([
        "🔗 Encoded Token Sequence",
        "📚 Learned Vocabulary ({})".format(len(learned_vocab)),
        "✅ Lossless Decompression Verification"
    ])
    
    with tab_encoded:
        st.subheader("BPE Encoded Token Representation")
        st.write("Below are the tokens produced by the tokenizer. Notice how multiple adjacent bytes have been compressed into singular higher-index tokens (> 255):")
        
        # Print list of tokens visually using custom hover chips
        chip_html = ""
        vocab_bytes = st.session_state.get('vocab_bytes', {})
        for tok in encoded_tokens:
            token_bytes = vocab_bytes.get(tok, b"")
            char_str = token_bytes.decode('utf-8', errors='replace')
            
            # Escape spaces for visible layout
            char_str_clean = char_str.replace(" ", "␣").replace("\n", "↵")
            chip_html += f'<div class="token-chip">{char_str_clean}<span class="token-chip-id">#{tok}</span></div>'
            
        st.markdown(f'<div class="glass-card" style="line-height: 2.2;">{chip_html}</div>', unsafe_allow_html=True)
        
        # Raw arrays
        col_enc_arr, col_dec_arr = st.columns(2)
        with col_enc_arr:
            st.markdown("#### Encoded Integer List:")
            st.code(encoded_tokens)
        with col_dec_arr:
            st.markdown("#### Raw UTF-8 Decoded Byte List:")
            st.code(list(decoded_bytes))
            
    with tab_vocab:
        st.subheader("BPE Merges & Learned Vocabulary Rules")
        st.write("The base vocabulary has 256 entries (UTF-8 byte values 0 to 255). Below are the new tokens that were learned dynamically from your text via BPE pairs:")
        
        if len(learned_vocab) == 0:
            st.info("No merges were performed. Increase the vocabulary size or supply longer text to trigger pair merging.")
        else:
            df_vocab = pd.DataFrame(learned_vocab)
            st.dataframe(
                df_vocab, 
                use_container_width=True,
                hide_index=True
            )
            
    with tab_decoded:
        st.subheader("Original vs. Decoded Text Check")
        
        is_identical = (text_input == decoded_text)
        
        if is_identical:
            st.success("🎉 Success! The decoded text matches the original input exactly (Lossless Tokenization).")
        else:
            st.error("❌ Mismatch! The decoded text does not match the original input text.")
            
        col_orig, col_dec = st.columns(2)
        with col_orig:
            st.markdown("**Original Text:**")
            st.text_area("Original", value=text_input, height=200, disabled=True, label_visibility="collapsed")
        with col_dec:
            st.markdown("**Decoded Text:**")
            st.text_area("Decoded", value=decoded_text, height=200, disabled=True, label_visibility="collapsed")
            
        st.markdown("#### Compression Stats Breakdown:")
        st.write(f"- Character count: `{len(text_input)}` chars ➡️ `{len(decoded_text)}` chars")
        st.write(f"- Token length: `{original_byte_len}` base bytes ➡️ `{encoded_token_len}` encoded tokens")
        st.write(f"- Percentage size: `{ (encoded_token_len / original_byte_len * 100):.1f}%` of original byte representation size")
