import json

file_path = '/home/wyq/桌面/mininet-RTC/newtrace/NY_4G_data_json/7Train_7BtrainNew.json'

with open(file_path, 'r') as f:
    data = json.load(f)

print("Downlink keys:", data['downlink'].keys())
print("Downlink content (first 100 chars):", str(data['downlink'])[:100])
