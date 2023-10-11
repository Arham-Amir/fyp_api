from flask import Flask, request
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from ultralytics import YOLO


def detection(path,model):
    results = model.predict(source=path)
    shape = results[0].boxes.orig_shape
    cls = results[0].boxes.cls
    xywh = results[0].boxes.xywh
    result_array = [shape, cls, xywh]
    print(result_array)
    return result_array

# Rest of the code remains the same
def calculate_midpoint(rect_width, rect_height):
    # Calculate midpoint
    midpoint_x = rect_width // 2
    midpoint_y = rect_height // 2
    return midpoint_x, midpoint_y

def determine_grid_cell(midpoint, grid_coordinates):
    # Determine the grid cell based on the midpoint
    for i, coord in enumerate(grid_coordinates):
        (x1, y1), (x2, y2) = coord
        if x1 <= midpoint[0] <= x2 and y1 <= midpoint[1] <= y2:
            return i + 1  # Grid cell numbering starts from 1
    return None

def calculate_grid_coordinates(image_width, image_height, num_rows, num_cols):
    grid_coordinates = []
    cell_width = image_width // num_cols
    cell_height = image_height // num_rows

    for i in range(num_rows):
        for j in range(num_cols):
            x1 = j * cell_width
            y1 = i * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            grid_coordinates.append(((x1, y1), (x2, y2)))

    return grid_coordinates

def position(data , path):
    # Load the image
    image_path = path # Replace with the actual image path
    image = Image.open(image_path)

    # Get image dimensions
    image_width, image_height = image.size


    # Calculate grid coordinates
    num_rows, num_cols = 3, 3  # 3x3 grid
    grid_coordinates = calculate_grid_coordinates(image_width, image_height, num_rows, num_cols)


    pos = []
    # rectangle wala kaam
    for box_val in data[2]:
      x, y, w, h = box_val
      left = int(x - w / 2)
      right = int(x + w / 2)
      top = int(y - h / 2)
      bottom = int(y + h / 2)

      if left < 0:
          left = 0
      if right > image_width - 1:
          right = image_width - 1
      if top < 0:
          top = 0
      if bottom > image_height - 1:
          bottom = image_height - 1

      # Calculate midpoint of the rectangle
      rect_midpoint = calculate_midpoint(right + left, bottom + top)

      # Determine the grid cell for the midpoint
      grid_cell = determine_grid_cell(rect_midpoint, grid_coordinates)
      pos.append(grid_cell)
      # Print the grid cell for the midpoint


    return pos


def main(img):
  count = 1
  path = img
  model = YOLO("yolov8n.pt")
  arr = detection(path,model)
  if(len(arr[1])):
    print(arr[0])
    pos = position(arr , path)
    print(pos)
  else:
    print("nothing detected")

app = Flask(__name__)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    print("Aooo Raja")
    file = request.files['frame']

    if file:
        file_data = file.read()
        file_size = len(file_data)

        # Just to show you can process the image, but not displaying it:
        npimg = np.frombuffer(file_data, np.uint8)

        # Decode the image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        #main(img)

        modified_image_path = 'modified_image.jpg'
        cv2.imwrite(modified_image_path, img)

        # Normally, you'd use OpenCV to process the image here

        print(f"Received frame size: {file_size} bytes")
        return {"message": "Frame received successfully!"}, 200
    else:
        return {"error": "No frame found!"}, 400

@app.route('/')
def hello_world():
    print('Hello World')
    return 'Hello World'

if __name__ == '__main__':
    app.run()
