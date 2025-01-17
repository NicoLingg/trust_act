import urllib.request
from pathlib import Path

WEIGHT_URLS = {
    'vit': {
        'url': 'https://www.cs.cmu.edu/~data4robotics/release/dataproj/vit_base/SOUP_1M_DH.pth',
        'path': 'weights/vit_base/SOUP_1M_DH.pth'
    },
    'resnet18': {
        'url': 'https://www.cs.cmu.edu/~data4robotics/release/dataproj/resnet18/IN_1M_resnet18.pth',
        'path': 'weights/resnet18/IN_1M_resnet18.pth'
    }
}

def download_file(url: str, dest: Path, desc: str = None):
    """Download file with progress bar"""
    def progress(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        print(f"\rDownloading {desc}: {percent}%", end="")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=progress if desc else None)
    print()

def download_weights(model_type: str):
    """Download pretrained weights if needed"""
    if model_type not in WEIGHT_URLS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        info = WEIGHT_URLS[model_type]
        dest = Path(info['path'])
        
        # Skip if already downloaded
        if dest.exists():
            print(f"Using existing weights for {model_type}")
            return dest
            
        # Download weights
        print(f"Downloading {model_type} weights...")
        download_file(info['url'], dest, model_type)
    except Exception as e:
        raise RuntimeError(f"Failed to download weights for {model_type}: {str(e)}")
    
    return dest


def main():
    download_weights('vit')
    download_weights('resnet18')
    
if __name__ == '__main__':
    main()