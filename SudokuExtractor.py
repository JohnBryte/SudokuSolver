# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:36:42 2019

@author: Johannes
"""

import cv2
import operator
import numpy as np
from MachineLearning import DigitRecognizer
from Helper import Helper
#from matplotlib import pyplot as plt

class SudokuExtractor:
    def __init__(self, image = None):
        if image != None:
            self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            image = None
        self.model = DigitRecognizer()
        self.model.load_model('num_reader')
        self.digits = None
        self.grid = None
        self.helper = Helper()
    
    def load_image(self, image_dst):
        """Loads image"""
        self.image = cv2.imread(image_dst, cv2.IMREAD_GRAYSCALE)
        
    def extract_puzzle(self):
        cropped_image = self.warp_image(self.find_grid(self.pre_process_image(self.image)))
        squares = self.helper.infer_grid(cropped_image)
        self.digits = self.get_digits(cropped_image, squares, 28)
        self.grid = self.get_grid()

    def pre_process_image(self, image, skip_dilate=False):
        """use blur, threshold and dilation to get the main features of the image"""
        
        # Gaussian blur with a kernal size (height, width) of 9.
        # Note that kernal sizes must be positive and odd and the kernel must be square.
        processed_img = cv2.GaussianBlur(image.copy(), (9, 9), 0)
        
        # Adaptive threshold using 11 nearest neighbour pixels
        processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    
        # Invert colours, so gridlines have non-zero pixel values.
        # Necessary to dilate the image, otherwise will look like erosion instead.
        processed_img = cv2.bitwise_not(processed_img, processed_img)
    
        if not skip_dilate:
            kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]],np.uint8)
            processed_img = cv2.dilate(processed_img, kernel)
        return processed_img
    
    def find_grid(self, processed_img):
        """find corners of largest contour which (hopefully) is the grid of the puzzle"""
        #mode CV_RETR_EXTERNAL retrieves only the extreme outer contours
        #method CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points
        new_image, contours, hierarchy = cv2.findContours(processed_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Sort contours from largest to smallest
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        grid = contours[0]
        
        #add each coordinate of (x,y) pair of the largest contour together
        #the index of the smallest sum is the top_left coordinate
        #the index of the maximum is the bottom_right
        br_tl_list = [point[0][0] + point[0][1] for point in grid]
        top_left = br_tl_list.index(min(br_tl_list))
        bottom_right = br_tl_list.index(max(br_tl_list))    
        
        #substract each coordinate of (x,y) pair of the largest contour
        #the index of the smallest difference is the bottom_left coordinate
        #the index of the maximum is the top_right     
        bl_tr_list = [point[0][0] - point[0][1] for point in grid]
        bottom_left = bl_tr_list.index(min(bl_tr_list))
        top_right = bl_tr_list.index(max(bl_tr_list))
        
        return [grid[top_left][0], grid[top_right][0], grid[bottom_right][0], grid[bottom_left][0]]
    
    def warp_image(self, proc_image):
        """Crops and warps a rectangular section from an image into a square of similar size."""
        # Rectangle described by top left, top right, bottom right and bottom left points
        top_left, top_right, bottom_right, bottom_left = proc_image[0], proc_image[1], proc_image[2], proc_image[3]
        
        # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
        src = np.float32([[top_left, top_right, bottom_right, bottom_left]])
        
        # Get the longest side in the rectangle
        side = max([
            self.helper.distance_between(bottom_right, top_right),
            self.helper.distance_between(top_left, bottom_left),
            self.helper.distance_between(bottom_right, bottom_left),
            self.helper.distance_between(top_left, top_right)
        ])
    
        # Describe a square with side of the calculated length, this is the new perspective we want to warp to
        dst = np.float32([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]])
        
        # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
        M = cv2.getPerspectiveTransform(src, dst)
        
        return cv2.warpPerspective(self.image, M, (int(side), int(side)))
    
    def cut_from_rect(self, img, rect):
        """Cuts a rectangle from an image using the top left and bottom right points."""
        """region_of_interest = img[y1:y2, x1:x2]"""
        return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]
    
    def find_largest_feature(self, inp_img, scan_tl=None, scan_br=None):
        """
        Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
        connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
        """
        img = inp_img.copy() # Copy the image, leaving the original untouched
        height, width = img.shape[:2]
        max_area = 0
        seed_point = (None, None) #Starting point
        
        if scan_tl is None:
            scan_tl = [0, 0]
    
        if scan_br is None:
            scan_br = [width, height]
            
        # Loop through the image
        for x in range(scan_tl[0], scan_br[0]):
            for y in range(scan_tl[1], scan_br[1]):
                # Only operate on light or white squares
                if img.item(y, x) == 255 and x < width and y < height: # Note that .item() appears to take input as y, x
                    area = cv2.floodFill(img, None, (x,y), 64)
                    if area[0] > max_area:
                        max_area = area[0]
                        seed_point = (x,y)

        # Colour everything remaining black
        for x in range(width):
            for y in range(height):
                if img.item(y,x) == 255 and x < width and y <  height:
                    cv2.floodFill(img, None, (x,y), 0)
        
        #[p is not None for p in seed_point] checks if (x,y) in seed_point are not None, but values
        if all([p is not None for p in seed_point]):
            cv2.floodFill(img, None, seed_point, 255)
        
        top, bottom, left, right = height, 0, width, 0
        #Find the bounding parameters
        for x in range(width):
            for y in range(height):
                if img.item(y,x) == 255:
                    top = y if y < top else top
                    bottom = y if y > bottom else bottom
                    left = x if x < left else left
                    right = x if x > right else right        
        #cv2.rectangle(img, (left, top), (right, bottom),(255, 0, 0))
        
        bounding_box = [(left, top), (right, bottom)]
        return bounding_box, seed_point
    
    def extract_digit(self, img, rect, size):
        """Extracts a digit (if one exists) from a Sudoku square."""
        digit_square = self.cut_from_rect(img, rect) # Get the digit box from the whole square
        
        # Use fill feature finding to get the largest feature in middle of the box
        # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
        h, w = digit_square.shape[:2]
        centre = int(np.mean([h, w]) / 2.5)
        #display_points(digit_square, [[centre, centre],[w - centre, h - centre]])
        bounding_box, seed = self.find_largest_feature(digit_square, [centre, centre], [w - centre, h - centre])
        digit = self.cut_from_rect(digit_square, bounding_box)
        
        w_b = bounding_box[1][0] - bounding_box[0][0]
        h_b = bounding_box[1][1] - bounding_box[0][1]
        if w_b > 0 and h_b > 0 and (w_b * h_b) > 100 and len(digit) > 0:
            return self.centralize_digit(digit, bounding_box, [h,w])
        else:
            return np.zeros((size, size), np.uint8)

    def image_centre(self, rectangle):
        h, w = rectangle.shape[:2]
        tl = (0,0)
        br = (w, h)
        x = (br[0] + tl[0]) / 2
        y = (br[1] + tl[1]) / 2
        centre = (x, y)
        return centre
    
    def centralize_digit(self, digit, bbox, size_of_dst):
        #Calcualte centre of digit_image
        digit_x, digit_y = self.image_centre(digit)[:2]
        
        #calculate center of dst
        dst = np.zeros((size_of_dst[0], size_of_dst[1]), np.uint8)
        dst_x, dst_y = self.image_centre(dst)

        ## (2) Calc offset
        x_offset = int(dst_x - digit_x)
        y_offset = int(dst_y - digit_y)
        
        ## (3) do slice-op `paste`
        h,w = digit.shape[:2]
    
        #cv2.rectangle(dst, (int(x_offset), int(y_offset)), (int(x_offset+w), int(y_offset+h)),(255, 0, 0))
        img = dst.copy()
        img[y_offset:y_offset+digit.shape[0], x_offset:x_offset+digit.shape[1]] = digit
        return cv2.resize(img, (28, 28))

    def get_digits(self, img, squares, size):
        """Extracts digits from their cells and build an array"""
        digits = []
        img = self.pre_process_image(img, skip_dilate = True)
        for square in squares:
            digits.append(self.extract_digit(img, square, size))
        return digits
    
    def get_grid(self):
        #self.model.load_model()
        #model = tf.keras.models.load_model('sudoku_num_reader.model')
        prediction = self.model.make_prediction(self.digits)
        grid = []
        row = []
        counter = 0
        for i in range(1, 82):    
            if(self.digits[i-1].any()):
                if prediction[counter] == 0:
                    row.append(' ')
                    counter += 1
                else:
                    row.append(prediction[counter])
                    counter += 1
            else:
                row.append(' ')
            if i != 0 and i%9==0 or i == 81:
                grid.append(row)
                row = []
        return grid

    def print_sudoku(self):
        print("+" + "---+"*9)
        for i, row in enumerate(self.grid):
            print(("|" + " {}   {}   {} |"*3).format(*[x if x != 0 else " " for x in row]))
            if i % 3 == 2:
                print("+" + "---+"*9)
            else:
                print("+" + "   +"*9)
                
    def show_extraction(self):
        processed = self.pre_process_image(self.image)
        corners = self.find_grid(processed)
        self.helper.display_points(processed, corners)
        cropped = self.warp_image(corners)
        self.helper.show_image(cropped)
        squares = self.helper.infer_grid(cropped)
        self.helper.display_rects(cropped, squares)
        self.digits = self.get_digits(cropped, squares, 28)
        self.helper.show_digits(self.digits)
        self.grid = self.get_grid()