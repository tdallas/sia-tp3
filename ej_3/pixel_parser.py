import pandas as pd

class Parser():
    def __init__(self):
        self.pixels = self.parse_pixels()

    def get_pixels(self):
        return self.pixels

    def parse_pixels(self):
        pixels_list = []
        with open('ej_3/pixels_map.txt') as f:
            lines = f.readlines() # list containing lines of file
            current_pixel = ''
            index_num = 0
            for line in lines:
                new_line = line.replace(' ', '').replace('\n', '')
                current_pixel += str(new_line) 
                index_num +=1
                if index_num == 7:
                    index_num = 0
                    pixels_list.append([current_pixel])
                    current_pixel = ''
        return pixels_list
