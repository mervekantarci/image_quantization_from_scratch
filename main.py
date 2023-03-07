from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import os
from _tkinter import TclError
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', required=True, action="store", type=str,
                        help='path to input file')
    parser.add_argument('--cluster_count', required=False, type=int, default=5,
                        help='total number of clusters (default 5)')
    parser.add_argument('--max_iter', required=False, type=int, default=10,
                        help='maximum number of iteration until convergence (default 10)')
    parser.add_argument('--random_init', required=False, action="store_true", default=False,
                        help='cluster centers will be randomly initialized. '
                             'Do not set this flag to choose colors from the interactive window')
    parser.add_argument('--save', required=False, action="store_true", default=False,
                        help='processed image will be saved')
    parser.add_argument('--no_display', required=False, action="store_true", default=False,
                        help='processed image will not be displayed')

    args_ = parser.parse_args()
    return args_


def compute_initial_centroids(img, num_clusters, random_init):
    tcl_error = False
    # Close the interactive window to use random initialization
    if not random_init:
        try:
            plt.imshow(img)
            clicked_points = plt.ginput(num_clusters, show_clicks=True)
            centroids = np.zeros((num_clusters,3))
            # width & height ordering should be reversed
            for i, point in enumerate(clicked_points):
                centroids[i] = img[int(point[1]), int(point[0]), :]
        except TclError:
            tcl_error = True
    if random_init or tcl_error:
        # when the interactive window is closed or args.random_init flag set true, initialize randomly
        centroids = np.random.uniform(0, 255, (num_clusters, 3))
    return centroids


def compute_centroids(data, clustered, centroids):
    new_centroids = np.zeros((centroids.shape[0],3))
    for cluster_no in range(centroids.shape[0]):
        # finds all values belongs to each cluster
        points_in_clusters = data[clustered == cluster_no]
        if points_in_clusters.shape[0] == 0:
            # some cluster centers might not be chosen by any points when initialized randomly
            new_centroids[cluster_no] = np.mean(centroids[cluster_no], axis=0)
        else:
            # compute new centroid
            new_centroids[cluster_no] = np.mean(points_in_clusters, axis=0)

    return new_centroids


def assign_to_cluster(data, centroids):
    distances = np.zeros((centroids.shape[0], data.shape[0]))
    for k in range(centroids.shape[0]):
        # compute distance between each image color in image and cluster centers
        distances[k, :] = np.linalg.norm(data - centroids[k], axis=1)
    # assign to the nearest cluster
    labels = np.argmin(distances, axis=0)

    return labels


def fit_clusters(data, initial_centroids, max_iter):
    centroids = initial_centroids
    for iter_no in range(max_iter):
        old_centroids = centroids
        clustered = assign_to_cluster(data, old_centroids)
        centroids = compute_centroids(data, clustered, old_centroids)
        # break in case of an early convergence
        if np.all(old_centroids == centroids):
            break
    return centroids


def quantize(img, K, max_iter=10, random_init=False):
    initial_colors = compute_initial_centroids(img, K, random_init)
    # flatten the image
    img_shaped = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    # find the final color values
    quantization_colors = fit_clusters(img_shaped, initial_colors, max_iter)
    # convert to int for the sake of visualization purposes
    quantization_colors = quantization_colors.astype(int)
    # assign colors to each pixel
    pixel_labels = assign_to_cluster(img_shaped, quantization_colors)
    quantized_img = quantization_colors[pixel_labels, :]

    # convert to 2d back
    quantized_img = np.reshape(quantized_img, (img.shape[0], img.shape[1], 3))

    return quantized_img


def output(img, save=True, show=True, filepath=None, cluster_count=0):
    if save:
        filename, file_extension = os.path.splitext(filepath)
        save_path = filename + "_quantized_" + str(cluster_count) + file_extension
        Image.fromarray(img.astype(np.uint8)).save(save_path)
    if show:
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    args = get_args()
    print(args)

    np.random.seed(5)  # for repeatable experiments
    im = cv2.imread(args.file)  # enter the path to the image
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    processed_img = quantize(im, K=args.cluster_count, max_iter=args.max_iter, random_init=args.random_init)
    output(processed_img, save=args.save, show=not args.no_display,
           filepath=args.file, cluster_count=args.cluster_count)





