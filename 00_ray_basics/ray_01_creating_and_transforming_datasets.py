# -*- coding: utf-8 -*-

import os, sys
import ray


# ----------
# REFERENCE
# https://docs.ray.io/en/latest/ray-overview/index.html#


# -----------------------------------------------------------------------------------------------------------
# Creating dataset
# -----------------------------------------------------------------------------------------------------------

# Create a Dataset of Python objects.
ds = ray.data.range(10000)

# -> Dataset(num_blocks=200, num_rows=10000, schema=<class 'int'>)
print(ds)


# ----------
# -> [0, 1, 2, 3, 4]
ds.take(5)

# <class 'int'>
ds.schema()


# -----------------------------------------------------------------------------------------------------------
# Creating dataset from Python objects, which are held as Arrow records
# -----------------------------------------------------------------------------------------------------------

ds = ray.data.from_items([
        {"sepal.length": 5.1, "sepal.width": 3.5,
         "petal.length": 1.4, "petal.width": 0.2, "variety": "Setosa"},
        {"sepal.length": 4.9, "sepal.width": 3.0,
         "petal.length": 1.4, "petal.width": 0.2, "variety": "Setosa"},
        {"sepal.length": 4.7, "sepal.width": 3.2,
         "petal.length": 1.3, "petal.width": 0.2, "variety": "Setosa"},
     ])


# ----------
# Dataset(num_blocks=3, num_rows=3,
#         schema={sepal.length: float64, sepal.width: float64,
#                 petal.length: float64, petal.width: float64, variety: object})
print(ds)

ds.show()
# -> {'sepal.length': 5.1, 'sepal.width': 3.5,
#     'petal.length': 1.4, 'petal.width': 0.2, 'variety': 'Setosa'}
# -> {'sepal.length': 4.9, 'sepal.width': 3.0,
#     'petal.length': 1.4, 'petal.width': 0.2, 'variety': 'Setosa'}
# -> {'sepal.length': 4.7, 'sepal.width': 3.2,
#     'petal.length': 1.3, 'petal.width': 0.2, 'variety': 'Setosa'}

ds.schema()
# -> sepal.length: double
# -> sepal.width: double
# -> petal.length: double
# -> petal.width: double
# -> variety: string


# -----------------------------------------------------------------------------------------------------------
# Datasets can be created from files on local disk or remote datasources such as S3.
#   - Any filesystem supported by pyarrow can be used to specify file locations.
#     You can also create a Dataset from existing data in the Ray object store
#     or Ray-compatible distributed DataFrames:
# -----------------------------------------------------------------------------------------------------------

# Create from CSV.
# Tip: "example://" is a convenient protocol to access the
# python/ray/data/examples/data directory.
ds = ray.data.read_csv("example://iris.csv")


# ----------
# Create from Parquet.
ds = ray.data.read_parquet("example://iris.parquet")


# -----------------------------------------------------------------------------------------------------------
# Datasets can be transformed in parallel using .map().
#   - Transformations are executed eagerly and block until the operation is finished.
#     Datasets also supports .filter() and .flat_map().
# -----------------------------------------------------------------------------------------------------------

import pandas

ds = ray.data.from_items([
        {"sepal.length": 5.1, "sepal.width": 3.5,
         "petal.length": 1.4, "petal.width": 0.2, "variety": "Setosa"},
        {"sepal.length": 4.9, "sepal.width": 3.0,
         "petal.length": 1.4, "petal.width": 0.2, "variety": "Setosa"},
        {"sepal.length": 4.7, "sepal.width": 3.2,
         "petal.length": 1.3, "petal.width": 0.2, "variety": "Setosa"},
     ])



# ----------
# Create 10 blocks for parallelism.
ds = ds.repartition(10)


# Find rows with sepal.length < 5.5 and petal.length > 3.5.
def transform_batch(df: pandas.DataFrame) -> pandas.DataFrame:
    return df[(df["sepal.length"] < 5.5) & (df["petal.length"] > 3.5)]


transformed_ds = ds.map_batches(transform_batch)

transformed_ds.show()




