import json
import sys

file_a = sys.argv[1]
file_b = sys.argv[2]
dict_a = None
dict_b = None

with open(file_a, "r") as file:
  dict_a = json.load(file)

with open(file_b, "r") as file:
  dict_b = json.load(file)

n = len(dict_a["labels"])
score_diff = 0
increase_labels = 0
decrease_labels = 0
same_labels = 0
same_labels = 0
same_scores = 0

for i in range(n):
  sa = dict_a["scores"][i]
  la = dict_a["labels"][i]
  sb = dict_b["scores"][i]
  lb = dict_b["labels"][i]

  if sa == sb:
    same_scores += 1
  else:
    score_diff += sb - sa
  if la == lb:
    same_labels += 1
  elif la == 1 or la == 4 and lb == 2:
    increase_labels += 1
  else:
    decrease_labels += 1

print(f"Same scores: {same_scores/n*100:.2f}%\nAverage score difference: {score_diff/n:.5f}\nSame labels: {same_labels/n*100:.2f}%\nIncreased labels: {increase_labels/n*100:.2f}%\nDecreased labels: {decrease_labels/n*100:.2f}%\n")
