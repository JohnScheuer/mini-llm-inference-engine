import struct
import os
try:
    from sentencepiece import SentencePieceProcessor
except ImportError:
    print("Instalando biblioteca...")
    os.system("pip install sentencepiece --break-system-packages")
    from sentencepiece import SentencePieceProcessor

input_model = "/mnt/e/ModelosLLM/tinyllama1.1B/tokenizer.model"
output_bin = "models/tokenizer.bin"

sp = SentencePieceProcessor(model_file=input_model)
vocab_size = sp.get_piece_size()

with open(output_bin, "wb") as f:
    f.write(struct.pack("i", 128)) 
    for i in range(vocab_size):
        score = sp.get_score(i)
        s = sp.id_to_piece(i).replace(' ', ' ')
        b = s.encode('utf-8')
        f.write(struct.pack("f", score))
        f.write(struct.pack("i", len(b)))
        f.write(b)
print(f"SUCESSO! Novo tokenizer criado com {vocab_size} tokens.")
