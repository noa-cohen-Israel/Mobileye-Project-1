try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.filters import minimum_filter
    from scipy.ndimage import zoom
    from PIL import Image
    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def convolve(kernel_path, img, gt):
    colored_kernel = np.asarray(Image.open(kernel_path)) / 255
    kernel_arr = np.mean(colored_kernel, axis=2)
    if gt:
        kernel_arr[kernel_arr > 0.6] = kernel_arr[kernel_arr > 0.6] * -1
    else:
        kernel_arr[kernel_arr < 0.6] = kernel_arr[kernel_arr < 0.6] * -1
    kernel = np.array(kernel_arr)
    return sg.convolve(img, kernel, mode='same')


def filter_and_convolve_by_size(img, orig_img, size):
    """
    :param img: image to convolve
    :param orig_img: the current image being filtered
    :param size: the size of the image to be convolved
    :return:
    """
    # zooming image according to "size"
    zoomed_img = zoom(img, size)

    # convolve - stage 1:
    convolved_img = convolve("part1/img1.jpg", zoomed_img, False)

    # convolve - stage 2:
    convolved_img = convolve("part1/img2.png", convolved_img, True)

    red_lights = []
    green_lights = []

    # filter - stage 1
    min_filter = minimum_filter(convolved_img, 30)
    lights = [(i, j) for i in range(0, len(min_filter)) for j in range(0, len(min_filter[0])) if
              min_filter[i][j] == convolved_img[i][j] and min_filter[i][j] < -500 and i > 5]

    # sorting by color
    for i, j in lights:
        pixel = orig_img[i][j]
        if max(pixel) == pixel[0]:  # red light should be the max pixel on first layer
            red_lights.append((i, j))
        elif max(pixel) == pixel[1]:  # green light should be the max pixel on second layer
            green_lights.append((i, j))

    # extracting the x's and y's of the red and green lights
    red_x, red_y = [red_light[0] for red_light in red_lights], [red_light[1] for red_light in red_lights]
    green_x, green_y = [green_light[0] for green_light in green_lights], [green_light[1] for green_light in
                                                                          green_lights]

    return np.array(red_y) * (1 / size), np.array(red_x) * (1 / size), np.array(green_y) * (1 / size), np.array(
        green_x) * (1 / size)


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and your imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    # opening image and converting to black and white
    color_img = np.asarray(Image.open(kwargs["image_path"])) / 255
    image_black_and_white = np.mean(color_img, axis=2)

    # sending smaller image to convolution for finding bigger tfls
    big_red_y_lights, big_red_x_lights, big_green_y_lights, big_green_x_lights, = filter_and_convolve_by_size(
        image_black_and_white,
        c_image, 0.5)

    # sending regular size image to convolution for finding regular size tfls
    reg_red_y_lights, reg_red_x_lights, reg_green_y_lights, reg_green_x_lights, = filter_and_convolve_by_size(
        image_black_and_white,
        c_image, 1)

    # returning bigger and smaller tfl candidates
    return np.append(big_red_y_lights, reg_red_y_lights), np.append(big_red_x_lights, reg_red_x_lights), np.append(
        big_green_y_lights, reg_green_y_lights), np.append(big_green_x_lights, reg_green_x_lights)


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, red_x, red_y, green_x, green_y, fig_num=None):
    # plt.figure(fig_num).clf()
    plt.imshow(image, cmap=plt.get_cmap(name='gray'))

    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()
    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)
    # plt.show()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))

    color_img = np.asarray(Image.open(image_path)) / 255
    image1 = np.mean(color_img, axis=2)

    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    red_x, red_y, green_x, green_y = find_tfl_lights(image, image_path=image_path, some_threshold=42)
    show_image_and_gt(image, objects, red_x, red_y, green_x, green_y, fig_num)

    plt.plot(red_x, red_y, 'ro', markersize=4)
    plt.plot(green_x, green_y, 'go', markersize=4)


def main(argv=None):
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = 'data/images/'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, 'dusseldorf_000049_0000*.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
