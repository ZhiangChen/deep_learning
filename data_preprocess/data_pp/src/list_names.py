#!/usr/bin/env python3
'''
Zhiang Chen
July,2016
'''
"Save the list of names in directory."

import os
wd = os.getcwd()
print("Current directory is \""+wd+"\"")
print("Start to get list of names in this directory? (yes/no)")
cmd = input()
assert cmd == "yes" or cmd == "no"
if cmd == "no":
	print("Input correct directory:")
	wd = input()
	assert os.path.isdir(wd)

files = os.listdir(wd)
images = list()
for name in files:
	debris = name.split('.')
	if debris[-1] == 'pcd':
		images.append(name)

file_name = wd+"/name_lists"
fo = open(file_name,'w')
file = ' '.join(images)
fo.write(file)
fo.close()
print("Saved!")
print("The number of images is "+str(len(images)))
print("The size of the saved file is %0.2fkB" % float(os.path.getsize(file_name)/1024))
