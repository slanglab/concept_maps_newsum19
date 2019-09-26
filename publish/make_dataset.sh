#!/usr/bin/env bash

# These two lines are just for record keeping

#split -b 50m publish/dataset.jsonl.zip x && mv x* publish
#split -b 50m publish/dataset_split.jsonl.zip y && mv y* publish

cat publish/x* > publish/dataset.jsonl.zip
unzip publish/dataset.jsonl.zip

cat publish/y* > publish/dataset_split.jsonl.zip
unzip publish/dataset_split.jsonl.zip
