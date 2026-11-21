# Data for the modeling benchmark

The training and test sets of the 7 image tasks, exactly as the runs used them. The 14 files (1.6 GB, too big for
plain git) are in the Hugging Face dataset [`tyranasaurus/WeightLiftingBench`](https://huggingface.co/datasets/tyranasaurus/WeightLiftingBench) under `model/data/`.
Get them with `bash fetch.sh <dir>`, which also checks `SHA256SUMS.txt`.

Each file is a NumPy `.npz` with `images` and `labels` (int64 class ids). Images are whole numbers 0 to 255 (uint8);
divide by 255 for the network input. The one exception is `digits-train.npz`, which already holds float32 values 0 to 1.

| Task | Train file | Test file | Classes | Source and split |
|---|---|---|---|---|
| digits | 60,000 x (1, 28, 28) | 10,000 x (28, 28) | 10 | MNIST official splits |
| clothing | 60,000 x (1, 28, 28) | 10,000 x (28, 28) | 10 | Fashion-MNIST official splits |
| cifar | 50,000 x (3, 32, 32) | 10,000 x (3, 32, 32) | 10 | CIFAR-10 official splits |
| letters | 27,455 x (1, 28, 28) | 7,172 x (28, 28) | 24 | Sign Language MNIST official CSVs; labels remapped to 0..23 (J and Z have no images) |
| traffic | 39,209 x (3, 32, 32) | 12,630 x (3, 32, 32) | 43 | GTSRB original release; whole image resized to 32 x 32 with PIL nearest |
| galaxies | 15,959 x (3, 128, 128) | 1,777 x (3, 128, 128) | 10 | Galaxy10 DECaLS; seeded stratified 80/10/10 split (seed 1234), train = train + val, 2 x 2 block mean to 128 |
| person | 20,000 x (3, 96, 96) | 10,961 x (3, 96, 96) | 2 | Visual Wake Words rule on COCO 2014; test = the reference model's held-out 10%; train = a seeded random 20,000 of the other 98,658 photos |

* The test files are the hidden test sets the scorer uses: `assemble_tasks.py` copies each one into its task folder as
  `tests/hidden.npz`, and each training file as `environment/data/train.npz`. The files' checksums match the ones the
  runs used.
* Each training set was rebuilt from the original recipe and checked: the rebuilt test set (letters, galaxies) is
  byte-identical to the sweep's, and the seeded draw of 10 examples per class reproduces the sweep's look-only examples
  (letters, traffic). Person's split reproduces the original cut points and counts. Details and code are in the
  `wsp-evals` repository (`experiments/write-vs-program/provenance/`, `tools/build_train.py`).
* Licences differ by source. COCO photos (person) are Flickr images under mixed Creative Commons licences, some
  non-commercial; Galaxy10 DECaLS is CC BY 4.0. Check the licences before sharing the data.
