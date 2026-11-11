# Data

The data is a table of English job postings from the Viet Nguyen AI machine learning course (`final_project.ods`, about 10 MB). It is not stored in this repository because the file is large.

## How to run the experiments

1. Put the file at `data/final_project.ods`, or set `JOB_DATA_PATH` to that file.
2. A `.csv` or `.csv.gz` file with the same columns also works:
   `title`, `location`, `description`, `function`, `industry`, `career_level`.

If you already have the course files on your computer, the script will also look for `final_project.ods` in the course folder.

The table has 8,074 rows and six columns. One row is dropped because the description is missing. Some career levels have very few examples. The smallest level has only four.
