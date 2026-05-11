import os
import urllib.request
from huggingface_hub import snapshot_download

def main():
    os.makedirs('weights/hf_cache', exist_ok=True)
    print("Downloading Grounding DINO weights...")
    snapshot_download('IDEA-Research/grounding-dino-base', cache_dir='weights/hf_cache')
    
    sam2_path = 'weights/sam2.1_hiera_large.pt'
    if not os.path.exists(sam2_path):
        print("Downloading SAM 2.1 weights...")
        urllib.request.urlretrieve(
            'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
            sam2_path
        )
        print("SAM 2.1 weights downloaded successfully.")
    else:
        print("SAM 2.1 weights already exist.")

if __name__ == '__main__':
    main()
