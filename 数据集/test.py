import json

with open('val_entity.json', 'r') as outfile:
    data = json.load(outfile)

print(data)

ans = 0
for i in data:
    if len(data[i])>1:
        ans += 1
print(ans)