# Data Pre-Process
 
### 1. Create name list file
Make sure the directory only contains pcd files (remove all backup hidden files)
```shell
$ roscd data_pp
$ cd src 
$ chmod +x list_names.py
$ ./list_names.py
```
Then a file named 'name_list' is created under the directory where the pcd files reside.
    
### 2. Get depth images
Change to the directory where pcd files reside.
```shell
$ rosrun data_pp data_pp
```
Then it will generate depth images in the directory where the pcd files reside. Users need to create a new folder and copy all bmp images into new folder.

### 3. Label the images
Make sure the diretory only contains bmp files (also remove all backup hidden files)
```shell
$ roscd data_pp
$ cd src
$ chmod +x label_images.py
$ ./label_images.py
```
It will crop all original depth image into 34x34 image and shift the cropped image to 9 positions.

### 4. Add noise
```shell
$ sudo apt-get install imagemagick
$ roscd data_pp
$ cd src
$ chmod +x add_noise.py
$ ./add_noise.py
```
It will randomly add noises to the cropped images, which is to solve the problem caused by the NAN from kinect camera and improve the generalization as well. Then it will save the all the cropped depth images with noise into a file, 'depth_data'

'depth_data' has 'dataset', 'names', 'faces' and 'orientations', that are saved as python dictionary:
```python
{
	'dataset': dataset,
	'names': names,
	'faces': faces,
	'orientations': orientations
}
```

The method of reading the file:
'''schange to the directory where 'depth_data' resides'''
```python3
from six.moves import cPickle as pickle
file_name = 'depth_data'
with open(file_name, 'rb') as f:
    save = pickle.load(f)
    dataset = save['dataset']
    names = save['names']
    orientations = save['orientations']
  	del save
```
Also refer to the example of imporint depth_data, 'import_data.py'
