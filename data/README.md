# Data

## CRISPR/Cas9 screens

The Achilles and Project SCORE CRISPR/Cas9 screen data were downloaded from the [DepMap data portal](https://depmap.org/portal/).
The data can be downloaded using the following command (from the root directory of the project).

```bash
make download_data
```

## Notes

Below are some notes on the data for future reference.

### Copy number

Below is a description of the copy number data values from a [post](https://forum.depmap.org/t/what-is-relative-copy-number-copy-number-ratio/104) on the DepMap community forum:

> Since we do not have matched normals, the output is a “copy ratio” or relative copy number.
> It is relative to the rest of the genome for that cell line.
> E.g. if the cell line is tetraploid we would not be able to see it from the relative copy number.
> These values are reported as log2(relative CN + 1) in the portal.

Therefore, to get the original relative copy number values, use the following transformation: `cn = (2^x) - 1`.
To be clear, the average value of the relative copy number is 1.
