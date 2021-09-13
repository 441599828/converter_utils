import PIL.Image as Image
import os
import numpy as np

def crop_image(im, new_height,new_width):
    width, height = im.size   

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    crop_im = im.crop((left, top, right, bottom)) #Cropping Image 

    return crop_im

def convert_inst(inst_file):
    with open(inst_file, 'r') as f:
        calib = f.readlines()
    P2 = np.array([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    P2=str(P2[:, 0:3].flatten()).strip('[').strip(']').replace('\n','')
    return P2
    
if __name__ == "__main__":

    file_root = "/home/whn/disk2/dataset/KITTI_dense_3DObj/data/training/depth/data_depth_selection/test_depth_completion_anonymous"

    image_root=os.path.join(file_root,'image')
    image_files=sorted(os.listdir(image_root))
    image_crop=os.path.join(file_root,'image_crop')

    velo_root=os.path.join(file_root,'velodyne_raw')
    velo_files=sorted(os.listdir(velo_root))
    velo_crop=os.path.join(file_root,'velodyne_raw_crop')

    inst_root=os.path.join(file_root,'calib')
    inst_files=sorted(os.listdir(inst_root))
    inst_convert=os.path.join(file_root,'intrinsics')

    for i in range(0,len(image_files)):
        img_file=os.path.join(image_root, image_files[i])
        img=Image.open(img_file)
        img_croped=crop_image(img, 352, 1216)
        img_croped.save(os.path.join(image_crop, str(i).zfill(6))+'.png')

        velo_file=os.path.join(velo_root, velo_files[i])
        velo=Image.open(velo_file)
        velo_croped=crop_image(velo, 352, 1216)
        velo_croped.save(os.path.join(velo_crop, str(i).zfill(6))+'.png')
        print('Saved: ', velo_file)

        inst_file=os.path.join(inst_root, inst_files[i])
        inst=convert_inst(inst_file)
        with open(os.path.join(inst_convert,str(i).zfill(6))+'.txt',"w") as f:
            f.write(inst)
        print("Finished: ",i)
