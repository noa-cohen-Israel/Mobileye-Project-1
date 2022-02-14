import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.models import load_model


def read_file(filename):
    with open(filename) as img_data:
        labels = json.load(img_data)['objects']
        return labels


def n_random_pixels(labels, n):
    random_labels = list(filter(lambda label: ['label'] != 'traffic light', labels))
    all_random_pixels = np.concatenate([l['polygon'] for l in random_labels])
    random_indices = np.random.choice(len(all_random_pixels), n, replace=False)
    return all_random_pixels[random_indices]


def find_center(polygon):
    return [x_center(polygon), y_center(polygon)]


def is_inside(x, y, polygon):
    left_up = False
    left_down = False
    right_up = False
    right_down = False
    for pix in polygon:
        if pix[0] <= x and pix[1] >= y:
            left_up = True
        if pix[0] <= x and pix[1] <= y:
            left_down = True
        if pix[0] >= x and pix[1] >= y:
            right_up = True
        if pix[0] >= x and pix[1] <= y:
            right_down = True
    return left_down and left_up and right_down and right_up


def n_lights(image, image_color, traffic_lights, n):
    res = []
    red_x, red_y, green_x, green_y = find_tfl_lights(image, colored_image=image_color, some_threshold=42)
    x_pixels = red_x + green_x
    y_pixels = red_y + green_y
    for i in range(len(x_pixels)):
        is_pix_inside = False
        for polygon in traffic_lights:
            if is_inside(x_pixels[i], y_pixels[i], polygon):
                is_pix_inside = True
        if not is_pix_inside:
            res.append([x_pixels[i], y_pixels[i]])
    return res[:n]


def x_center(polygon):
    min_x = min(list(map(lambda p: p[0], polygon)))
    max_x = max(list(map(lambda p: p[0], polygon)))
    return min_x + (max_x - min_x) // 2


def y_center(polygon):
    min_y = min(list(map(lambda p: p[1], polygon)))
    max_y = max(list(map(lambda p: p[1], polygon)))
    return min_y + (max_y - min_y) // 2


def parse_image(image_path):
    image_color = Image.open(image_path)
    image_gray = image_color.convert('L')
    image = np.array(image_gray) / 255

    image_color = np.array(image_color) / 255
    return image, image_color


def classify_pixels(image_path, labels):
    traffic_lights = list(filter(lambda label: label['label'] == 'traffic light', labels))
    polygons = list(map(lambda t: t['polygon'], traffic_lights))
    traffic_lights = list(map(find_center, polygons))
    image, image_color = parse_image(image_path)
    lights = n_lights(image, image_color, polygons, len(traffic_lights) // 2)
    non_traffic_lights = n_random_pixels(labels, len(traffic_lights) - len(lights))
    return traffic_lights, non_traffic_lights


def deal_with_edge_cases(center_index, image, size=81):
    center_index = [int(center_index[0]), int(center_index[1])]
    half_size = size // 2
    zeros_array = np.zeros((size, size, 3))
    difference_x = half_size - (center_index[0])
    difference_y = half_size - (center_index[1])
    for i in range(len(image)):
        for j in range(len(image[0])):
            zeros_array[i + difference_x][j + difference_y] = image[i][j]
    return zeros_array


def crop_image(img_path, pixels):
    images = []
    im = np.asarray(Image.open(img_path))

    for pixel in pixels:  # pixels is the array of light sources found in part1
        left = max(0, pixel[0] - 40)
        right = min(2048, pixel[0] + 40)
        top = max(0, pixel[1] - 40)
        bottom = min(1024, pixel[1] + 40)
        cropped = im[int(top):int(bottom) + 1, int(left):int(right) + 1, :]

        width, height = cropped.shape[:2]
        if width < 81 or height < 81:
            new_pixel = (pixel[0] - left, pixel[1] - top)

            cropped = deal_with_edge_cases(new_pixel[::-1], cropped)
        images.append(cropped)

    return images


def crop_around_tfls(img, tfl_pixels, not_tfl_pixels):
    labels = [1] * len(tfl_pixels) + [0] * len(not_tfl_pixels)
    images = crop_image(img, tfl_pixels + list(not_tfl_pixels))

    return images, labels


def convert_to_bin(image, img_name, is_bin):
    with open(f'{img_name}.bin', 'ab') as labels_bin:
        for i, im in enumerate(image):
            try:
                labels_bin.write(im.tobytes())
            except:
                labels_bin.write(im.to_bytes(1, "little"))


def create_bin_files(images, labels, path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    convert_to_bin(images, f"{path}/data", False)
    convert_to_bin(labels, f"{path}/labels", True)


def create_data_set():
    counter = 0
    for r, d, f in os.walk('./gtFine/val'):
        for file in f:
            if file.endswith(".json"):
                img = f'./leftImg8bit/val/{file.split("_")[0]}/' + file[:-20] + 'leftImg8bit.png'
                tfl_pixels, not_tfl_pixels = classify_pixels(read_file(f'./gtFine/val/{file.split("_")[0]}/' + file))
                counter += (len(tfl_pixels) + len(not_tfl_pixels))
                create_bin_files(*crop_around_tfls(img, tfl_pixels, not_tfl_pixels), "Data_dir/val")
    print('images in val: {}'.format(counter))

    for r, d, f in os.walk('./gtFine/train'):
        for file in f:
            if file.endswith(".json"):
                img = f'./leftImg8bit/train/{file.split("_")[0]}/' + file[:-20] + 'leftImg8bit.png'
                tfl_pixels, not_tfl_pixels = classify_pixels(read_file(f'./gtFine/train/{file.split("_")[0]}/' + file))
                counter += (len(tfl_pixels) + len(not_tfl_pixels))
                create_bin_files(*crop_around_tfls(img, tfl_pixels, not_tfl_pixels), "Data_dir/train")
    print('images in train: {}'.format(counter))


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)


# root = './'  #this is the root for your val and train datasets
data_dir = './Data_dir/'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}


# viz_my_data(num=(6, 6), **datasets['val'])


def crop_image_and_verify_tfl(img_path, tfl_pixels):
    """
    runs the machine on the light sources
    :param img_path: path of image to put in the machine
    :param tfl_pixels: pixels of light sources found in part1
    :return: list of percentages of each light source on the picture
    """
    # cropping around light sources
    images = crop_image(img_path, tfl_pixels)

    # verifying light sources
    loaded_model = load_model("part2/model.h5")
    images = np.asarray(images)
    l_predictions = loaded_model.predict(images)
    return l_predictions
