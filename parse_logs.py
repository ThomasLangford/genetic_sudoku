"""Convert text logs to a csv file."""

import csv
import os
from itertools import zip_longest

log_dir = "./logs"
folders = ["all_false", "dual", "elitism", "elitism_duel",
           "elitism_multi_mutate", "elitism_multi_mutate_duel", "mutli_mutate"]

for folder in folders:
    print(folder)
    folder_path = os.path.join(log_dir, folder)
    log_array = []
    name = folder + ".csv"
    csv_name = os.path.join(folder_path, name)
    for f in os.listdir(folder_path):
        if f.endswith(".txt"):
            log_array.append(f)
    all_trails = []
    for f_name in log_array:
        print(f_name)
        all_trails.append([None]+[i for i in range(1, 1001)])
        trail_list = []
        with open(os.path.join(folder_path, f_name)) as text_f:
            for line in text_f:
                if line[0] == "G":
                    trail_list.append(line.split()[2])
                elif "File:" in line:
                    trail_list.append(line)
                elif "found" in line.lower():
                    all_trails.append(trail_list)
                    trail_list = []
        all_trails.append([])
    with open(csv_name, "w", newline="") as out_csv:
        writer = csv.writer(out_csv)
        for values in zip_longest(*all_trails):
            writer.writerow(values)
    print("\n")
