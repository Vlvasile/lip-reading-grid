from pathlib import Path   
import zipfile            
import requests            
from tqdm import tqdm      


BASE_URL = "https://zenodo.org/records/3625687/files"

RAW_DIR = Path("data/raw")


def download_file(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)   
    path = RAW_DIR / name                         

    if path.exists():                             
        print(f"{name}: already downloaded, skipping")
        return path

    url = f"{BASE_URL}/{name}?download=1"
    tmp = path.with_suffix(path.suffix + ".part") 

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()                      
        total = int(r.headers.get("content-length", 0))  
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=name) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  
                f.write(chunk)
                bar.update(len(chunk))

    tmp.rename(path)  
    return path

def unzip(path: Path) -> None:
    marker = path.with_suffix(".extracted")
    if marker.exists():
        return
    with zipfile.ZipFile(path) as z:
                
        members = [m for m in z.namelist() if not m.startswith("__MACOSX")]
        z.extractall(RAW_DIR, members=members)
    marker.touch()


if __name__ == "__main__":
    speakers = [1, 2, 3, 4]
    names = ["alignments.zip"] + [f"s{i}.zip" for i in speakers]
    for name in names:
        unzip(download_file(name))