import urllib.request
import sys

url = "https://huggingface.co/karpathy/tiny-llamas/resolve/main/stories1B.bin?download=true"
output = "stories1B.bin"

def progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rBaixando: {percent}% [{count * block_size / 1e6:.1f}MB / {total_size / 1e6:.1f}MB]")
    sys.stdout.flush()

print("Iniciando download do Stories 1.1B (2.1 GB)...")
urllib.request.urlretrieve(url, output, reporthook=progress)
print("\nDownload concluído!")
