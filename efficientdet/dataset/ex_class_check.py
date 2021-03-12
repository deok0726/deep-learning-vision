import json
import os

with open("ex_annotation.json", "r") as ex_json:
    ex_check = json.load(ex_json)

for i in range(len(ex_check)):
    image_id.append(ex_check[i]['image_id'])

print(ex_check[0]['id'])