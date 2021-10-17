# Hospital Data Extract/Transform/Load Pipeline

This project creates an Extract/Transform/Load (ETL) pipeline that processes data related to the effectiveness of various hospitals.

## Installation

1. Clone this repository.
```bash
git clone https://github.com/tteague19/data-pipeline.git
```
2. Download the following datasets and store them in a directory called *data*
    * [Hospital General Information](https://data.cms.gov/provider-data/dataset/xubh-q36u)
    * [Timely and Effective Care - Hospital](https://data.cms.gov/provider-data/dataset/yv7e-xc69)
3. Install [Docker](https://docs.docker.com/get-docker/)
4. Create a  Docker image.
```bash
docker build -f docker/dockerfile -t IMAGE_NAME .
```
5. Run the `etl_pipeline.py` script in the container based on the image.
```bash
docker run --rm -v FULL_PATH_TO_LOCAL_REPO/data-pipeline:/project IMAGE_NAME python etl_pipeline.py
```

This set of instruction will create a database within a directory called *databases* that we can query to gain insights into the data.

## Motivation

The specific goal of this project is to answer two questions via the data available at [Hospital General Information](https://data.cms.gov/provider-data/dataset/xubh-q36u) and [Timely and Effective Care - Hospital](https://data.cms.gov/provider-data/dataset/yv7e-xc69).
1. For hospitals with `OP_31 >= 50` (the percentage of patients who had cataract surgery and had improvement in visual function within 90 days following the surgery), what are those hospitals' overall ratings?
2. Which state had the highest number of patients who left the emergency department before being seen? (`OP_22`)?

### General Details and Assumptions

I drop all columns that are not relevant to the two questions we posed above.

### Question 1 Details and Assumptions

In the data on hospitals, each hospital has rating of either 1, 2, 3, 4, or 5 with 1 corresponding to a poor rating and 5 a high rating. Unfortunately, the vast majority of hospitals in the dataset do not possess this score, and, in that case, the entry for the hospital contains a "Not Available" statement in place of a numeric rating. Given the large amount of missing data, an imputation of, for example, a rounded average of the scores that are present will lead to misleading conclusions. To answer other questions, though, it is unwise to remove all entries with missing data, though. Thus, I replaced each "Not Available" value for entries with a measurement of `OP_31` with a dummy value (0) that has the same data type as a score but cannot occur as a valid score.

### Question 2 Details and Assumptions

Similarly, a large number of entries that correspond to a measurement of `OP_22` are missing a value, which corresponds to the number of patients who left the emergency department before being seen. I adopted the same approach, this time using a dummy value of -1 since zero and positive integers are valid values for an amount of people.

## Development

This project could be improved in a multitude of ways. First, we
could expand the set of questions in which we are interested in
answering. Doing so would require the modification of the transformations function, which would further require investigation into the nature of valid data in, for example, the score column. To that end, I would like to add various checks to ensure the integrity of the existing data. I have, thus far, performed ad hoc verifications of the existing data at the time of writing, but a systematic approach that would automatically apply as the data changes would be advantageous. Finally, portions of the code could be refactored to assist in clarity.