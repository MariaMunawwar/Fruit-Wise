import requests
import binascii

url = 'http://127.0.0.1:5000/predict'
image_path = '/Users/mariakhan/banana.jpeg'

with open(image_path, 'rb') as img:
    files = {'file': img}
    response = requests.post(url, files=files)
    try:
        response_json = response.json()
        print("Detected fruits:", response_json.get("detected_fruits"))

        hex_data = response_json.get("image")
        if hex_data:
            img_bytes = binascii.unhexlify(hex_data)
            with open("output_image.jpg", "wb") as img_file:
                img_file.write(img_bytes)
            print("Image saved as output_image.jpg")
    except Exception as e:
        print("Error:", e)
        print("Response content:", response.content)
