#!/bin/bash

errant_parallel -orig input.txt -cor correct.txt -out ../annotations/gold.m2
errant_parallel -orig input.txt -cor baseline.txt -out ../annotations/baseline.m2
errant_parallel -orig input.txt -cor tune.txt -out ../annotations/tune.m2

echo "Baseline results" > data/annotations/eval.txt

errant_compare -hyp data/annotations/baseline.m2 -ref data/annotations/gold.m2 >> data/annotations/eval.txt

echo "Fine-tuning results" >> data/annotations/eval.txt

errant_compare -hyp data/annotations/tune.m2 -ref data/annotations/gold.m2 >> data/annotations/eval.txt






