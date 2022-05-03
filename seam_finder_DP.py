import numpy as np
import cv2 

def seam_finder(image, draw=True, i = 0) :

    pixel_energies = calculate_energy_of_image(image)
    lowest_seam_energy_map = calculate_lowest_seam_energies_map(pixel_energies)
    x_coordinate = calculate_lowest_seam_energies(lowest_seam_energy_map)

    if draw :
        seam_image = draw_seam(image, x_coordinate)

        cv2.imwrite('./seam'+str(i)+'.jpg', seam_image)
        return x_coordinate

    return x_coordinate

def draw_seam(image, x_coordinate) :

    for row in range(image.shape[0]) :
        image[row][x_coordinate[row]] = [0, 0, 255]

    return image

def calculate_lowest_seam_energies(lowest_seam_energy_map) :

    x_coordinate = []
    lowest_seam_energy_map = lowest_seam_energy_map[::-1]
    lowest_seam_energy_map = np.array(lowest_seam_energy_map)

    seam_energies_row = np.array(lowest_seam_energy_map[0])
    
    x = np.argmin(seam_energies_row)
    x_coordinate.append(x)

    x_left = max(x-1, 0)
    x_right = min(x+1, len(seam_energies_row)-1)

    for y in range(1, len(lowest_seam_energy_map)) :
        seam_energies_row = lowest_seam_energy_map[y]
        
        if min(seam_energies_row[x_left], seam_energies_row[x], seam_energies_row[x_right]) == seam_energies_row[x_left] :
            x = x_left
            x_left = max(x-1, 0)
            x_right = min(x+1, len(seam_energies_row)-1)
        elif min(seam_energies_row[x_left], seam_energies_row[x], seam_energies_row[x_right]) == seam_energies_row[x] :
            x = x
            x_left = max(x-1, 0)
            x_right = min(x+1, len(seam_energies_row)-1)
        else :
            x = x_right
            x_left = max(x-1, 0)
            x_right = min(x+1, len(seam_energies_row)-1)
        
        x_coordinate.append(x)

    x_coordinate = x_coordinate[::-1]

    return x_coordinate

def calculate_lowest_seam_energies_map(pixel_energies) :

    previous_seam_energies_row = list(pixel_energies[0])
    
    lowest_seam_energy_map = []
    lowest_seam_energy_map.append(previous_seam_energies_row)

    for y in range(1, len(pixel_energies)) :
        pixel_energies_row = pixel_energies[y]
        seam_energies_row = []

        for x, pixel_energy in enumerate(pixel_energies_row) :
            x_left = max(x-1, 0)
            x_right = min(x+1, len(pixel_energies_row)-1)
            x_range = range(x_left, x_right+1)

            min_seam_energy = pixel_energy + min(previous_seam_energies_row[x_i] for x_i in x_range)
            seam_energies_row.append(min_seam_energy)
        
        previous_seam_energies_row = seam_energies_row
        lowest_seam_energy_map.append(previous_seam_energies_row)

    return lowest_seam_energy_map

def calculate_energy_of_image(image) :
    
    image = image.astype('float32')

    energy = np.zeros((image.shape[0], image.shape[1]))

    for row in range(image.shape[0]) :
        for column in range(image.shape[1]) :
            if (image[row][column] == np.array([0, 0, 0])).all() :
                delta_x = 10000
                delta_y = 10000
            elif row == 0 and column == 0 :
                delta_x = abs(image[row][column][0] - image[row][column+1][0]) + abs(image[row][column][1] - image[row][column+1][1]) + abs(image[row][column][2] - image[row][column+1][2])
                delta_y = abs(image[row][column][0] - image[row+1][column][0]) + abs(image[row][column][1] - image[row+1][column][1]) + abs(image[row][column][2] - image[row+1][column][2])
            elif row == 0 and column == image.shape[1] - 1 :
                delta_x = abs(image[row][column-1][0] - image[row][column][0]) + abs(image[row][column-1][1] - image[row][column][1]) + abs(image[row][column-1][2] - image[row][column][2])
                delta_y = abs(image[row][column][0] - image[row+1][column][0]) + abs(image[row][column][1] - image[row+1][column][1]) + abs(image[row][column][2] - image[row+1][column][2])

            elif row == image.shape[0] - 1 and column == 0 :
                delta_x = abs(image[row][column][0] - image[row][column+1][0]) + abs(image[row][column][1] - image[row][column+1][1]) + abs(image[row][column][2] - image[row][column+1][2])
                delta_y = abs(image[row-1][column][0] - image[row][column][0]) + abs(image[row-1][column][1] - image[row][column][1]) + abs(image[row-1][column][2] - image[row][column][2])
            elif row == image.shape[0] - 1 and column == image.shape[1] - 1 :
                delta_x = abs(image[row][column-1][0] - image[row][column][0]) + abs(image[row][column-1][1] - image[row][column][1]) + abs(image[row][column-1][2] - image[row][column][2])
                delta_y = abs(image[row-1][column][0] - image[row][column][0]) + abs(image[row-1][column][1] - image[row][column][1]) + abs(image[row-1][column][2] - image[row][column][2])

            elif row == image.shape[0] - 1 :
                delta_x = abs(image[row][column-1][0] - image[row][column+1][0]) + abs(image[row][column-1][1] - image[row][column+1][1]) + abs(image[row][column-1][2] - image[row][column+1][2])
                delta_y = abs(image[row-1][column][0] - image[row][column][0]) + abs(image[row-1][column][1] - image[row][column][1]) + abs(image[row-1][column][2] - image[row][column][2])
            elif column == image.shape[1] - 1 :
                delta_x = abs(image[row][column-1][0] - image[row][column][0]) + abs(image[row][column-1][1] - image[row][column][1]) + abs(image[row][column-1][2] - image[row][column][2])
                delta_y = abs(image[row-1][column][0] - image[row+1][column][0]) + abs(image[row-1][column][1] - image[row+1][column][1]) + abs(image[row-1][column][2] - image[row+1][column][2])

            elif row == 0 :
                delta_x = abs(image[row][column-1][0] - image[row][column+1][0]) + abs(image[row][column-1][1] - image[row][column+1][1]) + abs(image[row][column-1][2] - image[row][column+1][2])
                delta_y = abs(image[row][column][0] - image[row+1][column][0]) + abs(image[row][column][1] - image[row+1][column][1]) + abs(image[row][column][2] - image[row+1][column][2])
            elif column == 0 :
                delta_x = abs(image[row][column][0] - image[row][column+1][0]) + abs(image[row][column][1] - image[row][column+1][1]) + abs(image[row][column][2] - image[row][column+1][2])
                delta_y = abs(image[row-1][column][0] - image[row+1][column][0]) + abs(image[row-1][column][1] - image[row+1][column][1]) + abs(image[row-1][column][2] - image[row+1][column][2])
                
            else :
                delta_x = abs(image[row][column-1][0] - image[row][column+1][0]) + abs(image[row][column-1][1] - image[row][column+1][1]) + abs(image[row][column-1][2] - image[row][column+1][2])
                delta_y = abs(image[row-1][column][0] - image[row+1][column][0]) + abs(image[row-1][column][1] - image[row+1][column][1]) + abs(image[row-1][column][2] - image[row+1][column][2])
            
            energy[row][column] = delta_x + delta_y

    return energy

def get_overlap_image(canvas1, canvas2) :

    overlap_mask = np.zeros_like(canvas1)

    x = []
    y = []

    for i in range(len(canvas2)) :
        for j in range(len(canvas2[0])) :
            if (canvas1[i][j] != np.array([0, 0, 0])).any() and (canvas2[i][j] != np.array([0, 0, 0])).any() :
                overlap_mask[i][j][:] = np.array([1, 1, 1])
                y.append(i)
                x.append(j)

    if len(x) != 0 and len(y) != 0 :
        return overlap_mask, [min(x), max(x), min(y), max(y)]
    else :
        return overlap_mask, []