import json
import os

def load_ins_id_to_instance_name(json_path):
    """
    Reads an episode_xxx.json file and
    returns a mapping dict from ins_id (int) to instance name (str).
    """
    with open(json_path, "r") as f:
        meta = json.load(f)

    # "ins_id_mapping" is stored as a string, so we need to parse it again.
    ins_map = json.loads(meta["ins_id_mapping"])

    id_to_name = {}
    for k_str, path in ins_map.items():
        # k_str is like "0", "1", "2", ... -> convert to int
        k = int(k_str)

        # Keep "background" and "unlabelled" as they are.
        if path in ("background", "unlabelled"):
            instance_name = path
        else:
            # e.g., "/World/scene_0/dice_269/base_link/visuals"
            parts = path.split("/")

            # Rule of thumb: tokens between "World", "scene_0", "base_link", "visuals"
            # are the instance names (e.g., dice_269, bed_jaysra_0)
            instance_name = None
            for p in parts:
                if not p:
                    continue
                if p in ("World", "scene_0", "base_link", "visuals"):
                    continue
                instance_name = p
                break

            # If the name is not found, use the full path.
            if instance_name is None:
                instance_name = path

        id_to_name[k] = instance_name

    return id_to_name


if __name__ == "__main__":
    json_path = "mapping/dataset/task-0021/episode_00210170/episode_00210170.json"   # Change to your file path
    id_to_name = load_ins_id_to_instance_name(json_path)
    
    for ins_id, name in id_to_name.items():
        print(f"Instance ID: {ins_id}, Name: {name}")

    # background_keywords = ["floor", "wall", "ceiling", "door", "window"]
    
    # background_ids = []
    # for ins_id, name in id_to_name.items():
    #     if any(keyword in name for keyword in background_keywords):
    #         background_ids.append(ins_id)

    # print("Instance IDs for background objects (floor, wall, ceiling, door, window):")
    # print(sorted(background_ids)) 