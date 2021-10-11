# Data Engineer Take Home Project (Task Duration 5 hours)

***

### Project overview

We're interested in creating an ETL pipeline that processes hospital related information. Other teams will query this
output to gain actionable insights. For this project, we will be working with 2 data sources:

1. [Hospital General Information](https://data.cms.gov/provider-data/dataset/xubh-q36u)
1. [Timely and Effective Care - Hospital](https://data.cms.gov/provider-data/dataset/yv7e-xc69)

Data dictionaries and dataset are both listed on the webpage. Please download them to your environment.

### Project requirements

The final project should contain these pieces:

1. An ETL process that will:
    * read in the data
    * apply transformations (if any)
    * handle erroneous data
    * handle schema evolution
    * save it to a storage layer
1. A storage layer that stores the processed data and be able to answer below sample questions:
    * For hospitals with `OP_31 >=50` (Percentage of patients who had cataract surgery and had improvement in visual
      function within 90 days following the surgery), what are those hospitals' overall ratings?
    * Which state has the highest number of patients who left the emergency department before being seen (`OP_22`)?
1. Runnable in any machine

### What we're looking for

1. Coding standards and best practices
1. Project organization
1. Scalable and maintainable production ready code

### Questions?

1. Feel free to make your own assumptions about things that are not clear. Be sure to document those assumptions.
1. If it's been over 5 hours, feel free to submit it and document what you would do to finish it.
1. If you're able to finish in less than 5 hours, feel free to add extra glitter and sparkles on top of the project
   (maybe dockerize the project).

***
