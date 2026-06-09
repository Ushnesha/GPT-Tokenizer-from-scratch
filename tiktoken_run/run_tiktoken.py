import tiktoken

#GPT-2
enc = tiktoken.get_encoding("gpt2")
text = "Hello,        my dog is cute"
encoded = enc.encode(text)
print(f"Spaces not merged in GPT2: {encoded}")


#GPT-4
enc = tiktoken.get_encoding("cl100k_base")
text = "Hello,        my dog is cute"
encoded = enc.encode(text)
print(f"Spaces merged in GPT4: {encoded}")

