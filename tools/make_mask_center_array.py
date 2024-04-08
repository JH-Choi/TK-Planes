import os
import json

label_folder = "/mnt/hdd/data/Okutama_Action/Chris_data/1.1.1/training_set/labels"

labels = os.listdir(label_folder)
labels = sorted(labels,key=lambda x: int(x.split("_")[1].split(".")[0]))

imgs = []
for idx,label in enumerate(labels):
    if idx % 2 != 0:
        continue
    lines = []
    with open("{}/{}".format(label_folder, label),"r") as f:
        lines = f.readlines()

    centers = []
    for line in lines:
        nums = line.split(" ")
        centers.append([float(nums[1]),float(nums[2])])
    imgs.append(centers)

d = json.dumps(imgs,indent=4)
with open("centers.json","w") as f:
    f.write(d)

