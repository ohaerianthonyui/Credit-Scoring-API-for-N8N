import requests

url = "http://127.0.0.1:5000/predict"
sample_input = {
    "Age": 33,
    "Sex": "male",
    "Job": 2,
    "Housing": "rent",
    "Saving accounts": "little",
    "Checking account": "<0",
    "Credit amount": 12000,
    "Duration": 24,
    "Purpose": "car"
}

response = requests.post(url, json=sample_input)
print("Status code:", response.status_code)
print("Response text:", response.text)

# Only parse JSON if status is 200
if response.status_code == 200:
    print(response.json())
else:
    print("Server error, cannot parse JSON")
