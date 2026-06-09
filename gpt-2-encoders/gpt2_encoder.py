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

for merge_str in bpe.split('\n'):
    print(merge_str)

