import numpy as np, glob, os

for path in sorted(set(
    glob.glob("data/training/unet/chips/*.npz") +
    glob.glob("data/training/unet/train/images/*.npz") +
    glob.glob("data/training/unet/train/masks/*.npz")
)):
    try:
        sz = os.stat(path).st_size
        if sz < 1000:
            print(f"TOO SMALL ({sz}B): {path}")
            os.remove(path)
        else:
            np.load(path, mmap_mode="r").close()
    except Exception as e:
        print(f"CORRUPTED ({sz}B): {path} -> {e}")
        os.remove(path)

print("Done!")
