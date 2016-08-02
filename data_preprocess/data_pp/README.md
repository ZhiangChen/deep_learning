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
$ sudo apt-get install imagemagick
$ roscd data_pp
$ cd src
$ chmod +x label_images.py
$ ./label_images.py
```
It will display every depth image, and require users to input the label according to the image. The labels should share same rule, like "toy_duck0", "toy_duck15", ... And it will generate different label values according to different labels. Same label shares same label value.

Then a file named 'depth_data' is created under the directory where the bmp files reside. It has 'dataset', 'labelset' and 'dictionary', that are saved as python dictionary:
```python
{
	'dataset': dataset,
	'labelset': labelset,
	'dictionary': dictionary,
}
```
'dictionary' contains the dictionary that maps label to label value.

The method of reading the file:
'''schange to the directory where 'depth_data' resides'''
```python3
from six.moves import cPickle as pickle
file_name = 'depth_data'
with open(file_name, 'rb') as f:
	save = pickle.load(f)
  	dataset = save['dataset']
  	labelset = save['labelset']
  	del save
```
