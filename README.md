# image_stitching

## How to use

```
python main.py imageFolder
```

# Requirement

opencv-python                       4.5.1.48 </br>
opencv-contrib-python               4.5.5.64 </br>
opencv-python-headless              3.4.17.61 </br>
numpy                               1.20.3 </br>
 </br>
Openv is used to load and save images and extract feature points.

## Result

Result image is saved in root folder
The file name is the same as the folder name

## Warning

- It is very inefficient code. It's just for studying.

 Especially, gain_compansation is inefficient.
 You can do it without that part.

In main.py, use 

```
stitching.image_stitching(fname_list, args.dir+'.jpg', NNDR=0.7, trial=500, compansation=False)
```

- I used image 320x240, 240x320

## Example data

Look demo_result folder
