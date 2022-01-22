import scipy

# try:
#     import os
#     import json
#     import glob
#     import argparse
#
#     import numpy as np
#     from scipy import signal as sg
#     from scipy.ndimage.filters import maximum_filter
#
#     from PIL import Image
#
#     import matplotlib.pyplot as plt
# except ImportError:
#     print("Need to fix the installation")
#     raise
#
#
# def get_cnvl(image, color):
#     #
#     print(image)
#
#     cn10 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0.40, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0.40, 0.55, 0.40, 0, 0, 0, 0,
#             0, 0, 0, 0.40, 0.55, 0.7, 0.55, 0.40, 0, 0, 0,
#             0, 0, 0.40, 0.55, 0.7, 15, 0.7, 0.55, 0.40, 0, 0,
#             0, 0.40, 0.55, 0.7, 15, 1, 15, 0.7, 0.55, 0.40, 0,
#             0, 0, 0.40, 0.55, 0.7, 15, 0.7, 0.55, 0.40, 0, 0,
#             0, 0, 0, 0.40, 0.55, 0.7, 0.55, 0.40, 0, 0, 0,
#             0, 0, 0, 0, 0.40, 0.55, 0.40, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0.40, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     # cnv3=[-2/9,-1/9,-1/9,0,8/9,0,-1/9,-1/9,-2/9,]
#     # cnv4=[-1/16,-1/16,-1/16,-1/16,-1/16,4/16,4/16,-1/16,4/16,4/16,-1/16,-1/16,-1/16,-1/16,-1/16,-1/16,]
#     # clr_k = Image.open("./leftImg8bit/val/frankfurt/kernel.png")
#     # grey_k = np.array(clr_k.convert("L"))
#     kernel = np.array(cn10)
#     cn10 = np.reshape(kernel, (11, 11))
#     # print(kernel)
#     # return sg.convolve(image,kernel,"same")
#     # highpass_filter_kernel = np.array([[-1 / 9, -1 / 9, -1 / 9],
#     #                                    [-1 / 9, 8 / 9, -1 / 9],
#     #                                    [-1 / 9, -1 / 9, -1 / 9]])
#
#  #    highpass_filter_kernel = [-1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, 1 / 10.8, 1 / 10.8, 1 / 10.8, 1 / 10.8, 1 / 10.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, 1 / 10.8, 1 / 10.8, 1 / 8, 1 / 16, 1 / 10.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, 1 / 10.8, 1 / 8, 1, 1 / 8, 1 / 10.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, 1 / 10.8, 1 / 8, 1 / 8, 1 / 8, 1 / 10.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, 1 / 10.8, 1 / 10.8, 1 / 10.8, 1 / 10.8, 1 / 10.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 26, -1 / 30.8, -1 / 30.8,
#  # -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 30.8, -1 / 34]
#
#     # highpass_filter_kernel = np.reshape(highpass_filter_kernel, (11, 11))
#     kernel = np.array([
#     -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.1, -0.1, -0.1,
#     -0.1, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 1, 0.8, 2, 0.8, 1, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 1, 2, 3, 2, 1, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 1, 0.8, 2, 0.8, 1, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.1, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
#     ])
#     mean = np.mean(kernel)
#     kernel=kernel-mean
#     kernel = np.reshape(kernel, (15, 15))
#     # plt.imshow(kernel)
#     # plt.show()
#     return sg.convolve2d(image, kernel, boundary="symm", mode='same')
#
#
# def find_tfl_lights(c_image: np.ndarray, **kwargs):
#     """
#     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
#     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
#     :param kwargs: Whatever config you want to pass in here
#     :return: 4-tuple of x_red, y_red, x_green, y_green
#     """
#     ### WRITE YOUR CODE HERE ###
#     ### USE HELPER FUNCTIONS ###
#     img = get_cnvl(image=c_image, color=0)
#     # plt.imshow(img, cmap=plt.get_cmap(name="gray"))
#     max_filter = maximum_filter(img, 50)
#     lights = [(i, j) for i in range(0, len(max_filter)) for j in range(0, len(max_filter[0])) if
#               max_filter[i][j] == img[i][j] and max_filter[i][j] > 150]
#     red_lights = []
#     green_lights = []
#     # print(kwargs['colored_image'])
#     colored_image = kwargs['colored_image']
#     print(colored_image.shape)
#     print(img.shape)
#     for i, j in lights:
#         # print(i, j)
#         pixel = colored_image[i][j]
#         if max(pixel) == pixel[0] and pixel[0]>pixel[1]+20 and pixel[0]>pixel[2]+20:
#             red_lights.append((i, j))
#         elif max(pixel) == pixel[1] and pixel[1]>pixel[0]+20 and pixel[1]>pixel[2]+20:
#             green_lights.append((i, j))
#
#     green_x, green_y = [green_light[0] for green_light in green_lights], [green_light[1] for green_light in green_lights]
#     red_x, red_y = [red_light[0] for red_light in red_lights], [red_light[1] for red_light in red_lights]
#
#     # convl = get_cnvl(c_image)
#     # plt.imshow(convl, cmap=plt.get_cmap(name="gray"))
#     return red_y, red_x, green_y,green_x
#
#
# ### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
# def show_image_and_gt(image, objs, fig_num=None):
#     plt.figure(fig_num).clf()
#     plt.imshow(image, cmap=plt.get_cmap(name="gray"))
#     labels = set()
#     if objs is not None:
#         for o in objs:
#             poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
#             plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
#             labels.add(o['label'])
#         if len(labels) > 1:
#             plt.legend()
#
#
# def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
#     """
#     Run the attention code
#     """
#     clr_image = Image.open(image_path)
#     grey_image = np.array(clr_image.convert("L"))
#
#     if json_path is None:
#         objects = None
#     else:
#         gt_data = json.load(open(json_path))
#         what = ['traffic light']
#         objects = [o for o in gt_data['objects'] if o['label'] in what]
#
#     show_image_and_gt(np.array(clr_image), objects, fig_num)
#
#     red_x, red_y, green_x, green_y = find_tfl_lights(grey_image, colored_image=np.array(clr_image), some_threshold=42)
#     plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
#     plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
#
#
# def main(argv=None):
#     """It's nice to have a standalone tester for the algorithm.
#     Consider looping over some images from here, so you can manually examine the results
#     Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
#     :param argv: In case you want to programmatically run this"""
#
#     parser = argparse.ArgumentParser("Test TFL attention mechanism")
#     parser.add_argument('-i', '--image', type=str, help='Path to an image')
#     parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
#     parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
#     args = parser.parse_args(argv)
#     default_base = './leftImg8bit/val/frankfurt'
#
#     if args.dir is None:
#         args.dir = default_base
#     flist = glob.glob(os.path.join(args.dir, 'frankfurt_000001_044787_leftImg8bit.png'))
#
#     for image in flist:
#         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
#
#         if not os.path.exists(json_fn):
#             json_fn = None
#         test_find_tfl_lights(image, json_fn)
#
#     if len(flist):
#         print("You should now see some images, with the ground truth marked on them. Close all to quit.")
#     else:
#         print("Bad configuration?? Didn't find any picture to show")
#     plt.show(block=True)
#
#
# if __name__ == '__main__':
#     main()


# try:
#     import os
#     import json
#     import glob
#     import argparse
#
#     import numpy as np
#     from scipy import signal as sg
#     from scipy.ndimage.filters import maximum_filter
#
#     from PIL import Image
#
#     import matplotlib.pyplot as plt
# except ImportError:
#     print("Need to fix the installation")
#     raise
#
#
# def get_cnvl(image):
#     kernel = np.array([
#     -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.1, -0.1, -0.1,
#     -0.1, -0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 1, 0.2, 1, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0.2, 3, 0.2, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 1, 0.2, 1, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1,
#     -0.1, -0.1, -0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.05, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.05, -0.1, -0.1, -0.1,
#     -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
#     ])
#     kernel = np.reshape(kernel, (15, 15))
#     return sg.convolve(image, kernel, mode="same")
#
#
# def find_tfl_lights(c_image: np.ndarray, **kwargs):
#     """
#     Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
#     :param c_image: The image itself as np.uint8, shape of (H, W, 3)
#     :param kwargs: Whatever config you want to pass in here
#     :return: 4-tuple of x_red, y_red, x_green, y_green
#     """
#     ### WRITE YOUR CODE HERE ###
#     ### USE HELPER FUNCTIONS ###
#     img = get_cnvl(c_image)
#     # plt.imshow(img, cmap=plt.get_cmap(name='gray'))
#
#     max_filter = maximum_filter(img, (30,50))
#     lights = [(i, j) for i in range(0, len(max_filter)) for j in range(0, len(max_filter[0])) if
#               max_filter[i][j] == img[i][j] and max_filter[i][j] > 2.6 and i>5]
#     red_lights = []
#     green_lights = []
#     colored_image = kwargs['colored_image']
#     for i, j in lights:
#         pixel = colored_image[i][j]
#         if max(pixel) == pixel[0]:
#             red_lights.append((i, j))
#         elif max(pixel) == pixel[1]:
#             green_lights.append((i, j))
#
#     red_x, red_y = [red_light[0] for red_light in red_lights], [red_light[1] for red_light in red_lights]
#     green_x, green_y = [green_light[0] for green_light in green_lights], [green_light[1] for green_light in
#                                                                           green_lights]
#
#     return red_y, red_x, green_y, green_x
#
#
# ### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
# def show_image_and_gt(image, objs, fig_num=None):
#     plt.figure(fig_num).clf()
#     plt.imshow(image, cmap=plt.get_cmap(name='gray'))
#     labels = set()
#     if objs is not None:
#         for o in objs:
#             poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
#             plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
#             labels.add(o['label'])
#         if len(labels) > 1:
#             plt.legend()
#
#
# def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
#     """
#     Run the attention code
#     """
#     image_color = Image.open(image_path)
#     image_gray = image_color.convert('L')
#     image = np.array(image_gray) / 255
#
#     image_color = np.array(image_color) /255
#
#     if json_path is None:
#         objects = None
#     else:
#         gt_data = json.load(open(json_path))
#         what = ['traffic light']
#         objects = [o for o in gt_data['objects'] if o['label'] in what]
#
#     show_image_and_gt(image_color, objects, fig_num)
#
#     red_x, red_y, green_x, green_y = find_tfl_lights(image, colored_image=image_color, some_threshold=42)
#     plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
#     plt.plot(green_x, green_y, 'ro', color='g', markersize=4)
#
#
# def main(argv=None):
#     """It's nice to have a standalone tester for the algorithm.
#     Consider looping over some images from here, so you can manually exmine the results
#     Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
#     :param argv: In case you want to programmatically run this"""
#
#     parser = argparse.ArgumentParser("Test TFL attention mechanism")
#     parser.add_argument('-i', '--image', type=str, help='Path to an image')
#     parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
#     parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
#     args = parser.parse_args(argv)
#     # default_base = './leftImg8bit/test/munich/'
#     default_base = './leftImg8bit/train/monchengladbach/'
#
#     if args.dir is None:
#         args.dir = default_base
#     # flist = glob.glob(os.path.join(args.dir, 'munster_000023_000019_leftImg8bit.png'))
#     flist = glob.glob(os.path.join(args.dir, 'monchengladbach_000000_009690_leftImg8bit.png'))
#     # flist = glob.glob(os.path.join(args.dir, 'frankfurt_000001_044787*.png'))
#
#     for image in flist:
#         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
#
#         if not os.path.exists(json_fn):
#             json_fn = None
#         test_find_tfl_lights(image, json_fn)
#
#     if len(flist):
#         print("You should now see some images, with the ground truth marked on them. Close all to quit.")
#     else:
#         print("Bad configuration?? Didn't find any picture to show")
#     plt.show(block=True)
#
#
# if __name__ == '__main__':
#     main()


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


def filter_and_convolve_by_size(img, orig_img, size):
    """
    :param img: image to convolve
    :param orig_img: the current image being filtered
    :param size: the size of the image to be convolved
    :return:
    """
    zoomed_img = zoom(img, size)  # zooming image according to "size"

    # convolve - stage 1:
    color_img = np.asarray(Image.open('./img1.jpg')) / 255
    array = np.mean(color_img, axis=2)
    array[array < 0.6] = array[array < 0.6] * -1
    kernel = np.array(array)
    convolved_img = sg.convolve(zoomed_img, kernel, mode='same')

    # convolve - stage 2:
    color_img = np.asarray(Image.open('./img2.png')) / 255
    array = np.mean(color_img, axis=2)
    array[array > 0.6] = array[array > 0.6] * -1
    kernel = np.array(array)
    convolved_img = sg.convolve(convolved_img, kernel, mode='same')

    red_lights = []
    green_lights = []

    # filter - stage 1
    min_filter = minimum_filter(convolved_img, 10)
    lights = [(i, j) for i in range(0, len(min_filter)) for j in range(0, len(min_filter[0])) if
              min_filter[i][j] == convolved_img[i][j] and min_filter[i][j] < -724 and i > 5]

    # filter - stage 2
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

    # sending smaller image to convolution for finding bigger tfls
    big_red_y_lights, big_red_x_lights, big_green_y_lights, big_green_x_lights, = filter_and_convolve_by_size(
        kwargs["image1"],
        c_image, 0.5)

    # sending regular size image to convolution for finding bigger tfls
    reg_red_y_lights, reg_red_x_lights, reg_green_y_lights, reg_green_x_lights, = filter_and_convolve_by_size(
        kwargs["image1"],
        c_image, 1)

    # returning bigger and smaller tfl candidates
    return np.append(big_red_y_lights, reg_red_y_lights), np.append(big_red_x_lights, reg_red_x_lights), np.append(
        big_green_y_lights, reg_green_y_lights), np.append(big_green_x_lights, reg_green_x_lights)


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image, cmap=plt.get_cmap(name='gray'))

    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


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
    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, image1=image1, some_threshold=42)

    plt.plot(red_x, red_y, 'ro', color='r', markersize=4)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = './leftImg8bit/test/munich/'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, 'munich_00028*'
                                             '.png'))

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
