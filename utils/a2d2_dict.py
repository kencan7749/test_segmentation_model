import json

#Image segmentation label for a2d2 dataset segmented other members

class LabelDict():
    def __init__(self, jfile=None):
        """
        jfile should be json file, which indicate label dict,
        but direct dict is also fine.
        """
        if type(jfile) is str:
            self.dict = json.load(jfile)
        elif type(jfile) is dict:
            self.dict = jfile
        else:
             #from 
             # https://github.com/open-mmlab/mmsegmentation/pull/175/files/722923584878157e7249c331a2c3279099640663)
             #It might be need additional transformation such as min -> 0 and max -> the unique number of the values
            self.dict= {
                        (255, 0, 0): 28, # Car 1
                        (200, 0, 0): 28, # Car 2
                        (150, 0, 0): 28, # Car 3
                        (128, 0, 0): 28, # Car 4
                        (182, 89, 6): 27, # Bicycle 1
                        (150, 50, 4): 27, # Bicycle 2
                        (90, 30, 1): 27, # Bicycle 3
                        (90, 30, 30): 27, # Bicycle 4
                        (204, 153, 255): 26, # Pedestrian 1
                        (189, 73, 155): 26, # Pedestrian 2
                        (239, 89, 191): 26, # Pedestrian 3
                        (255, 128, 0): 30, # Truck 1
                        (200, 128, 0): 30, # Truck 2
                        (150, 128, 0): 30, # Truck 3
                        (0, 255, 0): 32, # Small vehicles 1
                        (0, 200, 0): 32, # Small vehicles 2
                        (0, 150, 0): 32, # Small vehicles 3
                        (0, 128, 255): 19, # Traffic signal 1
                        (30, 28, 158): 19, # Traffic signal 2
                        (60, 28, 100): 19, # Traffic signal 3
                        (0, 255, 255): 20, # Traffic sign 1
                        (30, 220, 220): 20, # Traffic sign 2
                        (60, 157, 199): 20, # Traffic sign 3
                        (255, 255, 0): 29, # Utility vehicle 1
                        (255, 255, 200): 29, # Utility vehicle 2
                        (233, 100, 0): 16, # Sidebars
                        (110, 110, 0): 12, # Speed bumper
                        (128, 128, 0): 14, # Curbstone
                        (255, 193, 37): 6, # Solid line
                        (64, 0, 64): 22, # Irrelevant signs
                        (185, 122, 87): 17, # Road blocks
                        (0, 0, 100): 31, # Tractor
                        (139, 99, 108): 1, # Non-drivable street
                        (210, 50, 115): 8, # Zebra crossing
                        (255, 0, 128): 34, # Obstacles / trash
                        (255, 246, 143): 18, # Poles
                        (150, 0, 150): 2, # RD restricted area
                        (204, 255, 153): 33, # Animals
                        (238, 162, 173): 9, # Grid structure
                        (33, 44, 177): 21, # Signal corpus
                        (180, 50, 180): 3, # Drivable cobblestone
                        (255, 70, 185): 23, # Electronic traffic
                        (238, 233, 191): 4, # Slow drive area
                        (147, 253, 194): 24, # Nature object
                        (150, 150, 200): 5, # Parking area
                        (180, 150, 200): 13, # Sidewalk
                        (72, 209, 204): 255, # Ego car
                        (200, 125, 210): 11, # Painted driv. instr.
                        (159, 121, 238): 10, # Traffic guide obj.
                        (128, 0, 255): 7, # Dashed line
                        (255, 0, 255): 0, # RD normal street
                        (135, 206, 255): 25, # Sky
                        (241, 230, 255): 15, # Buildings
                        (96, 69, 143): 255, # Blurred area
                        (53, 46, 82): 255, # Rain dirt
                    }
