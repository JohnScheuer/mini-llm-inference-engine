import struct
from sentencepiece import SentencePieceProcessor

# CAMINHO DO SEU DRIVE E:
input_model = "/mnt/e/ModelosLLM/tinyllama1.1B/tokenizer.model"
output_bin = "models/tokenizer.bin"

sp = SentencePieceProcessor(model_file=input_model)
vocab_size = sp.get_piece_size()

with open(output_bin, "wb") as f:
    # O motor C++ espera o max_token_length inicial
    f.write(struct.pack("i", 128))
    
    for i in range(vocab_size):
        score = sp.get_score(i)
        s = sp.id_to_piece(i)
        s = s.replace(' ', ' ') # Espaço do Llama
        
        # Converte hex-tokens <0x0A> para bytes reais
        if s.startswith("<0x") and len(s) == 6:
            try: s = chr(int(s[3:5], 16))
            except: pass
                
        b = s.encode('utf-8')
        f.write(struct.pack("f", score))
        f.write(struct.pack("i", len(b)))
        f.write(b)

print(f"Sucesso! Tokenizer sincronizado com {vocab_size} tokens.")