# download the below encoder.json and vocab.bpe files
# !wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/vocab.bpe
# !wget https://openaipublic.blob.core.windows.net/gpt-2/models/1558M/encoder.json

import os, json

# Get path relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'encoder.json'), 'r') as f:
    enc = json.load(f)

with open(os.path.join(script_dir, 'vocab.bpe'), 'r') as f:
    bpe = f.read()

bpe_merges = {}
for idx, merge_str in enumerate(bpe.split('\n')):
    bpe_merges[idx] = tuple(merge_str.split())

print(f"Length of enc: {len(enc)}") #50257 = 256 base tokens + 50K merges + 1 special token '<|endoftext|>'
'''
<|endoftext|> means that we've reached the end of the text and it can wipe the context memory whatever came before
'''
print(f"Length of bpe: {len(bpe_merges)}")
