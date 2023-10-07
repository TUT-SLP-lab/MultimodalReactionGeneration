import os
import argparse
import json


def set_args(parser: argparse.ArgumentParser):
    parser.add_argument("--target", type=str, required=True)


def get_args():
    parser = argparse.ArgumentParser()
    set_args(parser)

    args = parser.parse_args()

    return args


def ex_name(file: str) -> str:
    person = file.rsplit(".", maxsplit=1)[0]
    person = person.split("_", maxsplit=1)[1:]
    person = "_".join(person)

    return person


def process():
    target = get_args().target

    person_info = {}

    rename_list = []
    for dirc in os.listdir(target):
        target_dirc = os.path.join(target, dirc)
        session_info = {
            "host": None,
            "comp": None,
            "host_no": None,
            "comp_no": None,
        }

        for file in os.listdir(target_dirc):
            file: str
            target_file = os.path.join(target_dirc, file)

            if "movie" == file[: len("movie")]:
                sub_name = "movie.mp4"
            elif "c-" == file[: len("c-")]:
                sub_name = "comp.wav"
                session_info["comp"] = ex_name(file)
            elif "h-" == file[: len("h-")]:
                sub_name = "host.wav"
                session_info["host"] = ex_name(file)
            elif "pair" == file[: len("pair")]:
                sub_name = "pair.wav"
            new_name = os.path.join(target_dirc, sub_name)
            rename_args = {
                "src": target_file,
                "dst": new_name,
            }
            rename_list.append(rename_args)

        if person_info.get(session_info["host"]) is None:
            person_info[session_info["host"]] = len(person_info)
        session_info["host_no"] = person_info[session_info["host"]]
        if person_info.get(session_info["comp"]) is None:
            person_info[session_info["comp"]] = len(person_info)
        session_info["comp_no"] = person_info[session_info["comp"]]

        info_file_name = os.path.join(target_dirc, "participant.json")
        with open(info_file_name, mode="w", encoding="utf-8") as f:
            infos = json.dumps(session_info, ensure_ascii=False)
            f.write(infos)
    info_file_name = os.path.join(target, "all_participant.json")
    with open(info_file_name, mode="w", encoding="utf-8") as f:
        infos = json.dumps(person_info, ensure_ascii=False)
        f.write(infos)

    for rename_args in rename_list:
        os.rename(**rename_args)


if __name__ == "__main__":
    process()
