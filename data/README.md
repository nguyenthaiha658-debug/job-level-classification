# Data

The dataset is a table of English-language job postings used in the
Viet Nguyen AI machine-learning course
(`final_project.ods`, about 10 MB). It is **not** committed to this
repository because of its size.

## How to run the experiments

1. Place the file at `data/final_project.ods`, **or** set the
   environment variable `JOB_DATA_PATH` to the file.
2. The loader also accepts `.csv` / `.csv.gz` with the same columns:
   `title`, `location`, `description`, `function`, `industry`, `career_level`.

If you already have the course materials unpacked, the scripts will
look for the original path as a fallback:

`final_project.ods` (file dữ liệu đi kèm khóa học)

The table has 8,074 rows and six columns. One row is dropped because
the description is missing. The target `career_level` is heavily
imbalanced (the smallest class has only four examples).
