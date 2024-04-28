#!/bin/bash

errant_parallel -orig data/knowledge/results/Input.txt -cor data/knowledge/results/Correct.txt -out data/knowledge/annotations/gold.m2
# errant_parallel -orig data/knowledge/results/Input.txt -cor baseline.txt -out ../annotations/baseline.m2
errant_parallel -orig data/knowledge/results/Input.txt -cor data/knowledge/results/Predicted.txt -out data/knowledge/annotations/distil.m2

# echo "Baseline results" > data/annotations/eval.txt

# errant_compare -hyp data/annotations/baseline.m2 -ref data/annotations/gold.m2 >> data/annotations/eval.txt

echo "Distillation results" >> data/knowledge/annotations/eval.txt

errant_compare -hyp data/knowledge/annotations/distil.m2 -ref data/knowledge/annotations/gold.m2 >> data/knowledge/annotations/eval.txt






