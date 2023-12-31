import os
import json
import argparse


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


def process():
    target = get_args().target

    person_info = {}
    person_appear = {}
    for mem in os.listdir(target):
        target_dirc = os.path.join(target, mem)
        if os.path.isfile(target_dirc):
            continue
        for file in os.listdir(target_dirc):
            if file[-4:] != "json":
                continue
            target_file = os.path.join(target_dirc, file)
            with open(target_file, mode="r", encoding="utf-8") as f:
                record = f.readline()
            dic = json.loads(record)

            person_info[dic["host"]] = dic["host_no"]
            person_info[dic["comp"]] = dic["comp_no"]
            person_appear[dic["host"]] = (
                1
                if person_appear.get(dic["host"]) is None
                else person_appear[dic["host"]] + 1
            )
            person_appear[dic["comp"]] = (
                1
                if person_appear.get(dic["comp"]) is None
                else person_appear[dic["comp"]] + 1
            )

    order_value = sorted(person_info.items(), key=lambda value: value[1])
    person_info = {}
    tmp_person_appear = {}
    for cell in order_value:
        person_info[cell[0]] = cell[1]
        tmp_person_appear[cell[0]] = person_appear[cell[0]]
    person_appear = tmp_person_appear

    record = json.dumps(person_info, ensure_ascii=False)
    path = os.path.join(target, "all_participant.json")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(record)

    record = json.dumps(person_appear, ensure_ascii=False)
    path = os.path.join(target, "all_participant_appear.json")
    with open(path, mode="w", encoding="utf-8") as f:
        f.write(record)


if __name__ == "__main__":
    process()
