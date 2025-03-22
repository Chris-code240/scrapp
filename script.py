import json

fields = []
with open('../processed_image.json', 'r') as file:

    json_data = json.load(file)


    for annotation in json_data[0]['annotations']:
        fields.append((annotation['label'], annotation['coordinates']))

with open("fields.json", "w") as file:
    file.write(json.dumps(fields))
