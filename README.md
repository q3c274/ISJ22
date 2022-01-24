Data and code for the paper **"Analytics on non-normalized data sources: more
learning, rather than more cleaning"**, submitted to Information Systems Journal.

### Requirements

- scikit-learn >= 0.22
- pandas
- joblib
- fastText pretrained model `wiki.en.bin` (8.5 Go, downloadable [here](https://fasttext.cc/docs/en/pretrained-vectors.html), section english, bin+txt).
This file must then be placed in the directory `code/fastText_bins`.

### Structure of the repository

- the folder `code` contains code to run the experiments, save the results and visualize them.
- results are stored in the folder `results`
- figures are stored in `latex/figures`
