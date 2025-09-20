import requests

def query(payload):
	headers = {
		"Accept" : "application/json",
		"Content-Type": "application/json"
	}
	response = requests.post(
		"https://cl1yyn4n4oicz0d9.us-east-1.aws.endpoints.huggingface.cloud",
		headers=headers,
		json=payload
	)
	return response.json()

output = query({
	"inputs": "Hello world!",
	"parameters": {}
}) 

print(output)