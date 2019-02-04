"""
Parse Output.

Usage:
    parse_output.py OUTPUT_PATH MATRIX_PATH [--straight-to-csv] [--round-to-csv] [--descriptive] [--frequency] [--demo-frequency] [--diag-frequency]
    parse_output.py (-h | --help)
    parse_output.py --version

Options:
    --straight-to-csv   Export the numpy matrix to csv as is
    --round-to-csv      Export the numpy matrix to csv rounding to 0 or 1.
    --descriptive       Prepare descriptive records for each patient/admission.
    --frequency         Condense descriptive records by frequency.
    --demo-frequency    Condense descriptive records by frequency of demographic data.
    --diag-frequency    Condense descriptive records by frequency of diagnoses.
    --h --help          Show this screen.
    --version           Show version.
"""

from __future__ import print_function

import csv
import pickle as pickle
import sys
from collections import namedtuple, Counter

import numpy as np
from docopt import docopt
from future import standard_library

# Import submodule
sys.path.append('icd9')
from icd9 import ICD9

standard_library.install_aliases()


def export_to_csv(path, table, header):
    with open(path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",", quoting=csv.QUOTE_MINIMAL)
        # Write header
        csv_writer.writerow(header)
        # Write content
        for row in table:
            csv_writer.writerow(row)


def create_descriptive_recordset(input_data, one_hot_map):
    # Create IC9 map of diagnoses codes & Record
    descriptors = {
        "admloc_": "admission_location",
        "disloc_": "discharge_location",
        "ins_": "insurance",
        "lang_": "language",
        "rel_": "religion",
        "mar_": "marital_status",
        "eth_": "ethnicity",
        "gen_": "gender",
        "dia_": "diagnoses"
    }
    enabled_descriptors = set()
    tree = ICD9('icd9\codes.json')
    diagnostic_codes_map = {}
    for key in one_hot_map.keys():
        for descriptor in descriptors.keys():
            if key.startswith(descriptor):
                enabled_descriptors.add(descriptors[key])
        if key.startswith("dia_"):
            try:
                condition = key[len("dia_D_"):]
                diagnostic_codes_map[key] = "{}-{}".format(condition,
                                                           tree.find(condition).description)
            except:
                diagnostic_codes_map[key] = key[len("dia_D_"):]
    DescriptiveRecord = namedtuple("DescriptiveRecord", list(sorted(enabled_descriptors)))

    # Actually process stuff
    for i, record in enumerate(input_data):

        record_lists = {key: [] for key in enabled_descriptors}

        # Process record
        for column in record:
            for column_mask, column_description in descriptors.items():
                if column.starts_with(column_mask):
                    if column_mask == "dia_":
                        condition = column[len("dia_D_"):]
                        condition_description = "{}-{}".format(condition,
                                                               diagnostic_codes_map[condition])
                        record_lists[column_description].append(condition_description)
                    else:
                        record_lists[column_description].append(column[len(column_mask):])

        # Package record
        packaged_record = [tuple(record_lists[key]) for key in sorted(enabled_descriptors)]

        sparse_descriptive_records.append(DescriptiveRecord(*packaged_record))

        # Report progress
        if i % 1000 == 0:
            print("{} records processed so far.".format(i + 1))

    return sparse_descriptive_records, list(sorted(enabled_descriptors))


if __name__ == '__main__':
    arguments = docopt(__doc__, version='medGAN - Parse Output 1.0')

    # Load data
    data = np.load("{}.npy".format(arguments["OUTPUT_PATH"]))
    with open("{}.types".format(arguments["MATRIX_PATH"]), "rb") as f:
        one_hot_map = pickle.load(f)

    # Reverse map
    index_map = {index: one_hot for one_hot, index in one_hot_map.items()}
    matrix_header = [index_map[index] for index in range(0, len(index_map))]

    if arguments["--straight-to-csv"]:
        export_to_csv("{}.csv".format(arguments["OUTPUT_PATH"]), data, header=matrix_header)

    if arguments["--round-to-csv"]:
        rounded_data = [[int(round(item)) for item in row] for row in data]

        export_to_csv(
            "{}.csv".format(arguments["OUTPUT_PATH"]),
            rounded_data,
            header=matrix_header
        )

    if arguments["--descriptive"] or arguments["--frequency"] or arguments["--demo-frequency"] \
            or arguments["--diag-frequency"]:
        cleaned_data = [[index_map[i] for i, item in enumerate(row) if int(round(item)) == 1]
                        for row in data]
        sparse_descriptive_records, descriptive_header = create_descriptive_recordset(data,
                                                                                      one_hot_map)

        if arguments["--descriptive"]:
            processed_descriptive_records = [[",".join(item) for item in record]
                                             for record in sparse_descriptive_records]
            export_to_csv(
                "{}_descriptive.csv".format(arguments["OUTPUT_PATH"]),
                processed_descriptive_records,
                header=descriptive_header
            )

        if arguments["--frequency"]:
            records_counter = Counter(sparse_descriptive_records)

            export_to_csv(
                "{}_frequency.csv".format(arguments["OUTPUT_PATH"]),
                [[count] + list(record) for record, count in records_counter.most_common()],
                header=["Count"] + descriptive_header
            )

        if arguments["--diag-frequency"]:
            filtered_records = [record.diagnoses for record in sparse_descriptive_records]
            records_counter = Counter(filtered_records)
            export_to_csv(
                "{}_diagnoses_frequency.csv".format(arguments["OUTPUT_PATH"]),
                [[count] + list(record) for record, count in records_counter.most_common()],
                header=["Count"] + descriptive_header
            )

        if arguments["--demo-frequency"]:
            filtered_records = [
                {field: record[field] for field in record._fields if field != "diagnoses"}
                for record in sparse_descriptive_records
            ]

            records_counter = Counter(filtered_records)
            export_to_csv(
                "{}_demo_frequency.csv".format(arguments["OUTPUT_PATH"]),
                [[count] + list(record.values())
                 for record, count in records_counter.most_common()],
                header=["Count"] + list(filtered_records[0].keys())
            )
