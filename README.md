# image_stitching

## How to use

```
python main.py imageFolder
```

## Result

Result image is saved in root folder
The file name is the same as the folder name

## Warning

It is very inefficient code. 
Just for study image stitching

Especially, gain_compansation is inefficient.
You can do it without that part.

In main.py, use 

```
stitching.image_stitching(fname_list, args.dir+'.jpg', NNDR=0.7, trial=500, compansation=False)
```

## Example data

Look demo_result folder
