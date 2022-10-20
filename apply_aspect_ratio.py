
import glob

original_data_folder = '../data/'
new_data_folder = '../data-resized/'

for file in list(glob.glob('*.txt')):
    print(file)