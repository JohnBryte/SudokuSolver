# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 13:21:51 2019

@author: Johannes
"""
import numpy as np
import cv2

class Helper:

    def show_image(self, img):
        """Shows an image until any key is pressed"""
        cv2.imshow('image', img)  # Display the image
        cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
        cv2.destroyAllWindows() # Close all windows

    def show_image_(self, img):
        """Shows an image until any key is pressed"""
        cv2.imshow('image', img)  # Display the image
        cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
        cv2.destroyAllWindows() # Close all windows
        
    def plot_many_images(self, images, titles, rows=1, columns=2):
        """Plots each image in a given list as a grid structure. using Matplotlib."""
        for i, image in enumerate(images):
            plt.subplot(rows, columns, i+1)
            plt.imshow(image, 'gray')
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])  # Hide tick marks
        plt.show()

    def display_points(self, in_img, points, radius=5, colour=(0, 0, 255)):
        """Draws circular points on an image."""
        img = in_img.copy()
        
        # Dynamically change to a colour image if necessary
        if len(colour) == 3:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for point in points:
            img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
        self.show_image(img)
        return img

    def display_rects(self, in_img, rects, colour=255):
        """Displays rectangles on the image."""
        img = in_img.copy()
        for rect in rects:
            img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
        self.show_image(img)
        return img

    def show_digits(self, digits, colour=255):
        """Shows list of 81 extracted digits in a grid format"""
        rows = []
        with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
        for i in range(9):
            row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
            rows.append(row)
        self.show_image(np.concatenate(rows))
    
    def distance_between(self, p1, p2):
        """Returns the scalar distance between two points"""
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))
    
    def infer_grid(self, cropped_img):
        """Infers 81 cell grid from a square image."""
        squares = []
        side = cropped_img.shape[:1]
        side = side[0] / 9
        for i in range(9):
            for j in range(9):
                p1 = (j * side, i * side)  # Top left corner of a bounding box
                p2 = ((j + 1) * side, (i + 1) * side)  # Bottom right corner of bounding box
                squares.append((p1, p2))
        return squares