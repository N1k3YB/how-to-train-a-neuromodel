import requests

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/upscale/conservative",
    headers={
        "authorization": f"Bearer sk-MYAPIKEY",
        "accept": "image/*"
    },
    files={
        "image": open("./low-res-flower.jpg", "rb"),
    },
    data={
        "prompt": "a flower",
        "output_format": "webp",
    },
)

if response.status_code == 200:
    with open("./flower.webp", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))

   # sk-xZKV5ulrbfyG3ii3cfdt2cfOhACJep1xAhJMapUzit6PhhIU
