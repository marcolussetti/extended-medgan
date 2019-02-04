"""
Process Mimic.

Usage:
    process_mimic.py ADMISSIONS_PATH DIAGNOSES_ICD_PATH OUTPUT_PATH [--count] [--full-icd9] [--admission] [--discharge] [--insurance] [--language] [--religion] [--marital] [--ethnicity] [--gender]
    process_mimic.py (-h | --help)
    process_mimic.py --version

Options:
  -h --help         Show this screen.
  --version         Show version.
  --full-icd9       Use full length ICD9 diagnostic codes.
  --count           Generate a count matrix rather than a binary matrix. Binary is default!
  --admission       Include in the matrix the admission location information (experimental).
  --discharge       Include in the matrix the discharge location information (experimental).
  --insurance       Include in the matrix the insurance information (experimental).
  --language        Include in the matrix the language information (experimental).
  --religion        Include in the matrix the religion information (experimental).
  --marital         Include in the matrix the marital status information (experimental).
  --ethnicity       Include in the matrix the ethnicity information (experimental).
  --gender          Include in the matrix the gender information (experimental).
"""

from __future__ import print_function

import csv
import itertools
import pickle as pickle
from collections import namedtuple
from datetime import datetime

import numpy as np
from docopt import docopt
from future import standard_library

standard_library.install_aliases()


def convert_to_icd9(diagnosis):
    if diagnosis.startswith('E'):
        if len(diagnosis) > 4:
            return diagnosis[:4] + '.' + diagnosis[4:]
        else:
            return diagnosis
    else:
        if len(diagnosis) > 3:
            return diagnosis[:3] + '.' + diagnosis[3:]
        else:
            return diagnosis


def convert_to_3digit_icd9(diagnosis):
    if diagnosis.startswith('E'):
        if len(diagnosis) > 4:
            return diagnosis[:4]
        else:
            return diagnosis
    else:
        if len(diagnosis) > 3:
            return diagnosis[:3]
        else:
            return diagnosis


def ingest_data(admission_path, diagnosis_path):
    with open(admission_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        AdmissionRecord = namedtuple(
            "AdmissionRecord",
            [field.lower() for field in csv_reader.__next__()]
        )

        admissions_list = [AdmissionRecord(*[field.lower() for field in line])
                           for line in csv_reader]

    with open(diagnosis_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',', quotechar='"')
        DiagnosisRecord = namedtuple(
            "DiagnosisRecord",
            [field.lower() for field in csv_reader.__next__()]
        )

        diagnosis_list = [DiagnosisRecord(*[field.lower() for field in line])
                          for line in csv_reader]

    assert len(admissions_list) > 0, "Empty admissions file, reset position"
    assert len(diagnosis_list) > 0, "Empty diagnosis file, reset position"

    return admissions_list, diagnosis_list


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Process Mimic 1.1')

    admission_path = arguments["ADMISSIONS_PATH"]
    diagnosis_path = arguments["DIAGNOSES_ICD_PATH"]
    output_path = arguments["OUTPUT_PATH"]

    # Ingest CSVs
    admissions_list, diagnosis_list = ingest_data(admission_path, diagnosis_path)

    # Extract types for demographic data
    if arguments["--admission"]:
        admission_locations = set([i.admission_location for i in admissions_list])
    if arguments["--discharge"]:
        discharge_locations = set([i.discharge_location for i in admissions_list])
    if arguments["--insurance"]:
        insurances = set([i.insurance for i in admissions_list])
    if arguments["--language"]:
        languages = set([i.language for i in admissions_list])
    if arguments["--religion"]:
        religions = set([i.religion for i in admissions_list])
    if arguments["--marital"]:
        marital_statuses = set([i.marital_status for i in admissions_list])
    if arguments["--ethnicity"]:
        ethnicities = set([i.ethnicity for i in admissions_list])

    print('Building pid-admission mapping, admission-date mapping')
    pid_admissions_map = {}
    admissions_date_map = {}

    for admission in admissions_list:
        pid = int(admission.subject_id)
        admission_id = int(admission.hadm_id)
        admission_time = datetime.strptime(admission.admittime, '%Y-%m-%d %H:%M:%S')
        admissions_date_map[admission_id] = admission_time
        if pid in pid_admissions_map:
            pid_admissions_map[pid].append(admission)
        else:
            pid_admissions_map[pid] = [admission]

    print('Building admission-dxList mapping')
    admissions_diagnosis_map = {}

    for diagnosis in diagnosis_list:
        admission_id = int(diagnosis.hadm_id)

        if arguments["--full-icd9"]:
            diagnosis_string = "D_" + convert_to_icd9(diagnosis.icd9_code[1:-1])
        else:
            diagnosis_string = "D_" + convert_to_3digit_icd9(diagnosis.icd9_code[1:-1])

        if admission_id in admissions_diagnosis_map:
            admissions_diagnosis_map[admission_id].append(diagnosis_string)
        else:
            admissions_diagnosis_map[admission_id] = [diagnosis_string]

    print('Building pid-sortedVisits mapping')
    pid_sorted_visits_map = {}

    for pid, admissions in pid_admissions_map.items():
        pid_sorted_visits_map[pid] = sorted(
            [(admissions_date_map[int(admission.hadm_id)], admissions_diagnosis_map[admission_id],
              admission)
             for admission in admissions]
        )

    print('Building pids, dates, strSeqs')
    pids = []
    dates = []
    seqs = []
    for pid, visits in pid_sorted_visits_map.items():
        pids.append(pid)
        seq = []
        date = []
        for visit in visits:
            date.append(visit[0])
            one_hot = []
            if arguments["--admission"]:
                one_hot.append("admloc_" + visit[2].admission_location)
            if arguments["--discharge"]:
                one_hot.append("disloc_" + visit[2].discharge_location)
            if arguments["--insurance"]:
                one_hot.append("ins_" + visit[2].insurance)
            if arguments["--language"]:
                one_hot.append("lang_" + visit[2].language)
            if arguments["--religion"]:
                one_hot.append("rel_" + visit[2].religion)
            if arguments["--marital"]:
                one_hot.append("mar_" + visit[2].marital_status)
            if arguments["--ethnicity"]:
                one_hot.append("eth_" + visit[2].ethnicity)
            one_hot.extend(
                ["dia_" + diagnosis for diagnosis in visit[1]])

            seq.append(one_hot)
        dates.append(date)
        seqs.append(seq)

    print('Creating types')
    # We'll concatenate all of the one-hot encodings for each category

    diagnoses = set(itertools.chain(*admissions_diagnosis_map.values()))
    types = {"dia_" + diagnosis: i for i, diagnosis in enumerate(diagnoses)}

    if arguments["--admission"]:
        admission_locations_offset = len(types)
        types.update({"admloc_" + location: i + admission_locations_offset
                      for i, location in enumerate(admission_locations)})
    if arguments["--discharge"]:
        discharge_locations_offset = len(types)
        types.update({"disloc_" + location: i + discharge_locations_offset
                      for i, location in enumerate(discharge_locations)})
    if arguments["--insurance"]:
        insurances_offset = len(types)
        types.update({"ins_" + insurance: i + insurances_offset
                      for i, insurance in enumerate(insurances)})
    if arguments["--language"]:
        languages_offset = len(types)
        types.update({"lang_" + language: i + languages_offset
                      for i, language in enumerate(languages)})
    if arguments["--religion"]:
        religions_offset = len(types)
        types.update({"rel_" + religion: i + religions_offset
                      for i, religion in enumerate(religions)})
    if arguments["--marital"]:
        marital_statuses_offset = len(types)
        types.update({"mar_" + marital_status: i + marital_statuses_offset
                      for i, marital_status in enumerate(marital_statuses)})
    if arguments["--ethnicity"]:
        ethnicities_offset = len(types)
        types.update({"eth_" + ethnicity: i + ethnicities_offset
                      for i, ethnicity in enumerate(ethnicities)})

    print('Converting strSeqs to intSeqs, and making types')
    new_sequences = []
    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                new_visit.append(types[code])
            new_patient.append(new_visit)
        new_sequences.append(new_patient)

    print('Constructing the matrix')

    patientsNumber = len(new_sequences)
    codesNumber = len(types)
    matrix = np.zeros((patientsNumber, codesNumber)).astype('float32')
    inverted_types = {v: k for k, v in types.items()}

    for i, patient in enumerate(new_sequences):
        for visit in patient:
            for code in visit:
                if arguments["--count"]:
                    matrix[i][code] += 1.
                else:
                    matrix[i][code] = 1.

    # Dump results
    pickle.dump(pids, open(output_path + '.pids', 'wb'), -1)
    pickle.dump(matrix, open(output_path + '.matrix', 'wb'), -1)
    pickle.dump(types, open(output_path + '.types', 'wb'), -1)
