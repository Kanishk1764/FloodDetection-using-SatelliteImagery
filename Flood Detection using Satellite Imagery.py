apikey='Z2d0ZTBsNzVwdXE4ZXFpcGF1OGFlcWI1YW86MTgxMDhjOTctOWMyNi00YTQ1LThlNzgtMWVjZmRjMmJhOTQy'
# write the config file
config_dict={'apikey': apikey, 'format_type': 'json', 'org': 'nvidia'}
with open('config', 'w') as f: 
    f.write(';WARNING - This is a machine generated file.  Do not edit manually.\n')
    f.write(';WARNING - To update local config settings, see "ngc config set -h"\n')
    f.write('\n[CURRENT]\n')
    for k, v in config_dict.items(): 
        f.write(k+'='+v+'\n')

# preview the config file
!cat config
# move the config file to ~/.ngc
!mkdir -p ~/.ngc & mv config ~/.ngc/
# login to NGC's docker registry
!docker login -u '$oauthtoken' -p $apikey nvcr.io
# unzip data file
!unzip data/flood_data.zip -d flood_data
# import dependencies
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from mpl_toolkits.basemap import Basemap

# set environment variables
%set_env LOCAL_DATA_DIR=/dli/task/flood_data

# set paths for images and masks
image_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images')
mask_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'masks')
# define function to count number of images per region
def count_num_images(file_dir): 
    """
    This function returns a dictionary representing the count of images for each region as the key. 
    """
    # list all files in the directory
    file_list=os.listdir(file_dir)
    region_count={}
    # iterate through the file_list and count by region
    for file_name in file_list: 
        region=file_name.split('_')[0]
        if (len(file_name.split('.'))==2) and (region in region_count): 
            region_count[region]+=1
        elif len(file_name.split('.'))==2: 
            region_count[region]=1
    return region_count
# count images and masks by region
images_count=count_num_images(os.path.join(image_dir, 'all_images'))
masks_count=count_num_images(os.path.join(mask_dir, 'all_masks'))

# display counts
print(f'-----number of images: {sum(images_count.values())}-----')
display(sorted(images_count.items(), key=lambda x: x[1]))

print(f'-----number of masks: {sum(masks_count.values())}-----')
display(sorted(masks_count.items(), key=lambda x: x[1]))
# define function to get coordinates from catalog
def get_coordinates(catalog_dir): 
    """
    This function returns a list of boundaries for every image as [[lon, lat], [lon, lat], [lon, lat], etc.] in the catalog. 
    """
    catalog_list=os.listdir(catalog_dir)
    all_coordinates=[]
    for catalog in catalog_list: 
        # check if it's a directory based on if file_name has an extension
        if len(catalog.split('.'))==1:
            catalog_path=f'{catalog_dir}/{catalog}/{catalog}.json'
            # read catalog
            with open(catalog_path) as f: 
                catalog_json=json.load(f)
            # parse out coordinates
            coordinates_list=catalog_json['geometry']['coordinates'][0]
            lon=[coordinates[0] for coordinates in coordinates_list]
            all_coordinates.append(lon)
            lat=[coordinates[1] for coordinates in coordinates_list]
            all_coordinates.append(lat)
    return all_coordinates
# set paths for images catalog
image_catalog_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'catalog', 'sen1floods11_hand_labeled_source')
image_coordinates_list=get_coordinates(image_catalog_dir)

# create figure
plt.figure(figsize=(15, 10))

# create a Basemap
m=Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

# display blue marble image
m.bluemarble(scale=0.2) # 0.2 downsamples to 1350x675 image
m.drawcoastlines(color='white', linewidth=0.2) # add coastlines
m.drawparallels(range(-90, 90, 10), labels=[0, 1, 0, 0], color='white', textcolor='black')
m.drawmeridians(range(-180, 180, 10), labels=[0, 0, 0, 1], color='white', textcolor='black', rotation=90)

# flatten lat and lon coordinate lists
image_lon=[image_coordinates_list[x] for x in range(len(image_coordinates_list)) if x%2==0]
image_lon=np.concatenate(image_lon).ravel()
image_lat=[image_coordinates_list[x] for x in range(len(image_coordinates_list)) if x%2==1]
image_lat=np.concatenate(image_lat).ravel()

# convert lon/lat to x/y map projection coordinates
x, y=m(image_lon, image_lat)
plt.scatter(x, y, s=10, marker='o', color='Red') 

plt.title('Data Distribution')
plt.show()
# define function to get extent of an image from catalog
def get_extent(file_path): 
    """
    This function returns the extent as [left, right, bottom, top] for a given image. 
    """
    # read catalog for image
    with open(file_path) as f: 
        catalog_json=json.load(f)
    coordinates=catalog_json['geometry']['coordinates'][0]
    coordinates=np.array(coordinates)
    # get boundaries
    left=np.min(coordinates[:, 0])
    right=np.max(coordinates[:, 0])
    bottom=np.min(coordinates[:, 1])
    top=np.max(coordinates[:, 1])
    return left, right, bottom, top
# define function to plot by region
def tiles_by_region(region_name, plot_type='images'): 
    # set catalog and images/masks path
    catalog_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'catalog', 'sen1floods11_hand_labeled_source')
    if plot_type=='images': 
        dir=os.path.join(image_dir, 'all_images')
        cmap='viridis'
    elif plot_type=='masks': 
        dir=os.path.join(mask_dir, 'all_masks')
        cmap='gray'
    else: 
        raise Exception('Bad Plot Type')

    # initiate figure boundaries, which will be modified based on the extent of the tiles
    x_min, x_max, y_min, y_max=181, -181, 91, -91
    fig=plt.figure(figsize=(15, 15))
    ax=plt.subplot(111)
    
    # iterate through each image/mask and plot
    file_list=os.listdir(dir)
    for each_file in file_list:
        # check if image/mask is related to region and a .png file
        if (each_file.split('.')[-1]=='png') & (each_file.split('_')[0]==region_name): 
            # get boundaries of the image
            extent=get_extent(f"{catalog_dir}/{each_file.split('.')[0]}/{each_file.split('.')[0]}.json")
            x_min, x_max=min(extent[0], x_min), max(extent[1], x_max)
            y_min, y_max=min(extent[2], y_min), max(extent[3], y_max)
            image=mpimg.imread(f'{dir}/{each_file}')
            plt.imshow(image, extent=extent, cmap=cmap)

    # set boundaries of the axis
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    plt.show()
tiles_by_region(region_name='Spain', plot_type='images')
# define function to plot by boundaries
def tiles_by_boundaries(left_top, right_bottom, plot_type='images'): 
    # set catalog and images/masks path
    catalog_dir=os.path.join(os.getenv('LOCAL_DATA_DIR'), 'catalog', 'sen1floods11_hand_labeled_source')
    if plot_type=='images': 
        dir=os.path.join(image_dir, 'all_images')
        cmap='viridis'
    elif plot_type=='masks': 
        dir=os.path.join(mask_dir, 'all_masks')
        cmap='gray'
    else: 
        raise Exception('Bad Plot Type')

    # initiate figure boundaries, which will be modified based on the extent of the tiles
    x_min, x_max, y_min, y_max=left_top[0], right_bottom[0], right_bottom[1], left_top[1]
    ax_x_min, ax_x_max, ax_y_min, ax_y_max=181, -181, 91, -91

    fig=plt.figure(figsize=(15, 15))
    ax=plt.subplot(111)

    # iterate through each image/mask and plot
    file_list=os.listdir(dir)
    for each_file in file_list: 
        # check if image/mask is a .png file
        if each_file.split('.')[-1]=='png': 
            # get boundaries of the image/mask
            extent=get_extent(f"{catalog_dir}/{each_file.split('.')[0]}/{each_file.split('.')[0]}.json")
            (left, right, bottom, top)=extent
            if (left>x_min) & (right<x_max) & (bottom>y_min) & (top<y_max):
                ax_x_min, ax_x_max=min(left, ax_x_min), max(right, ax_x_max)
                ax_y_min, ax_y_max=min(bottom, ax_y_min), max(top, ax_y_max)
                image=mpimg.imread(f'{dir}/{each_file}')
                plt.imshow(image, extent=extent, cmap=cmap)

    # set boundaries of the axis
    ax.set_xlim([ax_x_min, ax_x_max])
    ax.set_ylim([ax_y_min, ax_y_max])
    plt.show()

tiles_by_boundaries(left_top=(-0.966, 38.4), right_bottom=(-0.597, 38.0), plot_type='images')
# import dependencies
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types

import warnings
warnings.filterwarnings("ignore")
batch_size=4

@pipeline_def
def simple_pipeline():
    # use fn.readers.file to read encoded images and labels from the hard drive
    pngs, labels=fn.readers.file(file_root=image_dir)
    # use the fn.decoders.image operation to decode images from png to RGB
    images=fn.decoders.image(pngs, device='cpu')
    # specify which of the intermediate variables should be returned as the outputs of the pipeline
    return images, labels
# create and build pipeline
pipe=simple_pipeline(batch_size=batch_size, num_threads=4, device_id=0)
pipe.build()
# run the pipeline
simple_pipe_output=pipe.run()

images, labels=simple_pipe_output
print("Images is_dense_tensor: " + str(images.is_dense_tensor()))
print("Labels is_dense_tensor: " + str(labels.is_dense_tensor()))
# define a function display images
def show_images(image_batch):
    columns=4
    rows=1
    # create plot
    fig=plt.figure(figsize=(15, (15 // columns) * rows))
    gs=gridspec.GridSpec(rows, columns)
    for idx in range(rows*columns):
        plt.subplot(gs[idx])
        plt.axis("off")
        plt.imshow(image_batch.at(idx))
    plt.tight_layout()

show_images(images)
import random

@pipeline_def
def augmentation_pipeline():
    # use fn.readers.file to read encoded images and labels from the hard drive
    image_pngs, _=fn.readers.file(file_root=image_dir)
    # use the fn.decoders.image operation to decode images from png to RGB
    images=fn.decoders.image(image_pngs, device='cpu')
    
    # the same augmentation needs to be performed on the associated masks
    mask_pngs, _=fn.readers.file(file_root=mask_dir)
    masks=fn.decoders.image(mask_pngs, device='cpu')
    
    image_size=512
    roi_size=image_size*.5
    roi_start_x=image_size*random.uniform(0, 0.5)
    roi_start_y=image_size*random.uniform(0, 0.5)
    
    # use fn.resize to investigate an roi, region of interest
    resized_images=fn.resize(images, size=[512, 512], roi_start=[roi_start_x, roi_start_y], roi_end=[roi_start_x+roi_size, roi_start_y+roi_size])
    resized_masks=fn.resize(masks, size=[512, 512], roi_start=[roi_start_x, roi_start_y], roi_end=[roi_start_x+roi_size, roi_start_y+roi_size])
    
    # use fn.resize to flip the image
    flipped_images=fn.resize(images, size=[-512, -512])
    flipped_masks=fn.resize(masks, size=[-512, -512])
    return images, resized_images, flipped_images, masks, resized_masks, flipped_masks
pipe=augmentation_pipeline(batch_size=batch_size, num_threads=4, device_id=0)
pipe.build()
augmentation_pipe_output=pipe.run()
# define a function display images
augmentation=['original', 'resized', 'flipped']
def show_augmented_images(pipe_output):
    image_batch, resized_image_batch, flipped_image_batch, mask_batch, resized_mask_batch, flipped_mask_batch=pipe_output
    columns=6
    rows=batch_size
    # create plot
    fig=plt.figure(figsize=(15, (15 // columns) * rows))
    gs=gridspec.GridSpec(rows, columns)
    grid_data=[image_batch, resized_image_batch, flipped_image_batch, mask_batch, resized_mask_batch, flipped_mask_batch]
    grid=0
    for row_idx in range(rows): 
        for col_idx in range(columns): 
            plt.subplot(gs[grid])
            plt.axis('off')
            plt.title(augmentation[col_idx%3])
            plt.imshow(grid_data[col_idx].at(row_idx))
            grid+=1
    plt.tight_layout()
show_augmented_images(augmentation_pipe_output)
show_augmented_images(pipe.run())
@pipeline_def
def rotate_pipeline():
    images, _=fn.readers.file(file_root=image_dir)
    masks, _=fn.readers.file(file_root=mask_dir)
    images=fn.decoders.image(images, device='cpu')
    masks=fn.decoders.image(masks, device='cpu')
    
    angle=fn.random.uniform(range=(-30.0, 30.0))
    rotated_images = fn.rotate(images.gpu(), angle=angle, fill_value=0, keep_size=True, device='gpu')
    rotated_masks = fn.rotate(masks.gpu(), angle=angle, fill_value=0, keep_size=True, device='gpu')
    
    return rotated_images, rotated_masks
pipe=rotate_pipeline(batch_size=batch_size, num_threads=4, device_id=0)
pipe.build()
rotate_pipe_output=pipe.run()
# define a function display images
def show_rotate_images(pipe_output):
    image_batch, rotated_batch=pipe_output
    columns=batch_size
    rows=2
    fig=plt.figure(figsize=(15, (15 // columns) * rows))
    gs=gridspec.GridSpec(rows, columns)
    grid_data=[image_batch.as_cpu(), rotated_batch.as_cpu()]
    grid=0
    for row_idx in range(rows): 
        for col_idx in range(columns): 
            plt.subplot(gs[grid])
            plt.axis('off')
            plt.imshow(grid_data[row_idx].at(col_idx))
            grid+=1
    plt.tight_layout()
show_rotate_images(rotate_pipe_output)


# set environment variables
import os
import json

%set_env KEY=my_model_key

%set_env LOCAL_PROJECT_DIR=/dli/task/tao_project
%set_env LOCAL_DATA_DIR=/dli/task/flood_data
%set_env LOCAL_SPECS_DIR=/dli/task/tao_project/spec_files
os.environ["LOCAL_EXPERIMENT_DIR"]=os.path.join(os.getenv("LOCAL_PROJECT_DIR"), "unet")

%set_env TAO_PROJECT_DIR=/workspace/tao-experiments
%set_env TAO_DATA_DIR=/workspace/tao-experiments/data
%set_env TAO_SPECS_DIR=/workspace/tao-experiments/spec_files
%set_env TAO_EXPERIMENT_DIR=/workspace/tao-experiments/unet

!mkdir $LOCAL_EXPERIMENT_DIR

# mapping up the local directories to the TAO docker
mounts_file = os.path.expanduser("~/.tao_mounts.json")

drive_map = {
    "Mounts": [
            # Mapping the data directory
            {
                "source": os.environ["LOCAL_PROJECT_DIR"],
                "destination": "/workspace/tao-experiments"
            },
            # Mapping the specs directory.
            {
                "source": os.environ["LOCAL_SPECS_DIR"],
                "destination": os.environ["TAO_SPECS_DIR"]
            },
            # Mapping the data directory.
            {
                "source": os.environ["LOCAL_DATA_DIR"],
                "destination": os.environ["TAO_DATA_DIR"]
            },
        ],
    "DockerOptions": {
        "user": "{}:{}".format(os.getuid(), os.getgid())
    }
}

# writing the mounts file
with open(mounts_file, "w") as mfile:
    json.dump(drive_map, mfile, indent=4)
# list all available models
!ngc registry model list nvidia/tao/pretrained_semantic_segmentation:*
# create directory to store the pre-trained model
!mkdir -p $LOCAL_EXPERIMENT_DIR/pretrained_resnet18/

# download the pre-trained segmentation model from NGC
!ngc registry model download-version nvidia/tao/pretrained_semantic_segmentation:resnet18 \
    --dest $LOCAL_EXPERIMENT_DIR/pretrained_resnet18
!tree -a tao_project/unet/pretrained_resnet18
# remove existing splits
!rm -rf $LOCAL_DATA_DIR/images/train
!mkdir -p $LOCAL_DATA_DIR/images/train
!rm -rf $LOCAL_DATA_DIR/images/val
!mkdir -p $LOCAL_DATA_DIR/images/val

!rm -rf $LOCAL_DATA_DIR/masks/train
!mkdir -p $LOCAL_DATA_DIR/masks/train
!rm -rf $LOCAL_DATA_DIR/masks/val
!mkdir -p $LOCAL_DATA_DIR/masks/val

# import dependencies
from random import sample
import shutil

# define split ratio
split=0.75

# get all images
file_list=os.listdir(f"{os.environ['LOCAL_DATA_DIR']}/images/all_images")
image_count=len(file_list)
train_image_list=sample(file_list, int(image_count*split))
val_image_list=[file for file in file_list if file not in train_image_list]

# move all training images to train directory
for each_file in train_image_list: 
    if each_file.split('.')[-1]=='png': 
        shutil.copyfile(f"{os.environ['LOCAL_DATA_DIR']}/images/all_images/{each_file}", f"{os.environ['LOCAL_DATA_DIR']}/images/train/{each_file}")
        shutil.copyfile(f"{os.environ['LOCAL_DATA_DIR']}/masks/all_masks/{each_file}", f"{os.environ['LOCAL_DATA_DIR']}/masks/train/{each_file}")

# move all validation images to val directory
for each_file in val_image_list: 
    if each_file.split('.')[-1]=='png': 
        shutil.copyfile(f"{os.environ['LOCAL_DATA_DIR']}/images/all_images/{each_file}", f"{os.environ['LOCAL_DATA_DIR']}/images/val/{each_file}")
        shutil.copyfile(f"{os.environ['LOCAL_DATA_DIR']}/masks/all_masks/{each_file}", f"{os.environ['LOCAL_DATA_DIR']}/masks/val/{each_file}")

# combining configuration components in separate files and writing into one
!cat $LOCAL_SPECS_DIR/resnet18/dataset_config.txt \
     $LOCAL_SPECS_DIR/resnet18/model_config.txt \
     $LOCAL_SPECS_DIR/resnet18/training_config.txt \
     > $LOCAL_SPECS_DIR/resnet18/combined_config.txt
!cat $LOCAL_SPECS_DIR/resnet18/combined_config.txt
# read the config file
!cat $LOCAL_SPECS_DIR/resnet18/dataset_config.txt
# combining configuration components in separate files and writing into one
!cat $LOCAL_SPECS_DIR/resnet18/dataset_config.txt \
     $LOCAL_SPECS_DIR/resnet18/model_config.txt \
     $LOCAL_SPECS_DIR/resnet18/training_config.txt \
     > $LOCAL_SPECS_DIR/resnet18/combined_config.txt
!cat $LOCAL_SPECS_DIR/resnet18/combined_config.txt
# train model
!tao unet train -e $TAO_SPECS_DIR/resnet18/combined_config.txt \
                -r $TAO_EXPERIMENT_DIR/resnet18 \
                -n resnet18 \
                -m $TAO_EXPERIMENT_DIR/pretrained_resnet18/pretrained_semantic_segmentation_vresnet18/resnet_18.hdf5 \
                -k $KEY
print('Model for every epoch at checkpoint_interval mentioned in the spec file:')
print('---------------------')
!tree -a $LOCAL_EXPERIMENT_DIR/resnet18
# evaluate the model using the same validation set as training
!tao unet evaluate -e $TAO_SPECS_DIR/resnet18/combined_config.txt\
                   -m $TAO_EXPERIMENT_DIR/resnet18/weights/resnet18.tlt \
                   -o $TAO_EXPERIMENT_DIR/resnet18/ \
                   -k $KEY
# remove any previous inference
!rm -rf $LOCAL_PROJECT_DIR/tao_infer_testing/*
# perform inference on the validation set
!tao unet inference -e $TAO_SPECS_DIR/resnet18/combined_config.txt \
                    -m $TAO_EXPERIMENT_DIR/resnet18/weights/resnet18.tlt \
                    -o $TAO_PROJECT_DIR/tao_infer_testing \
                    -k $KEY
# import dependencies
import matplotlib.pyplot as plt
import random

# define simple grid visualizer
def visualize_images(num_images=10):
    overlay_path=os.path.join(os.environ['LOCAL_PROJECT_DIR'], 'tao_infer_testing', 'vis_overlay_tlt')
    inference_path=os.path.join(os.environ['LOCAL_PROJECT_DIR'], 'tao_infer_testing', 'mask_labels_tlt')
    actual_path=os.path.join(os.environ['LOCAL_DATA_DIR'], 'masks', 'val')
    inference_images_path=os.path.join(os.environ['LOCAL_DATA_DIR'], 'images', 'val')
        
    fig_dim=4
    fig, ax_arr=plt.subplots(num_images, 4, figsize=[4*fig_dim, num_images*fig_dim], sharex=True, sharey=True)
    ax_arr[0, 0].set_title('Overlay')
    ax_arr[0, 1].set_title('Input')
    ax_arr[0, 2].set_title('Inference')
    ax_arr[0, 3].set_title('Actual')
    ax_arr[0, 0].set_xticks([])
    ax_arr[0, 0].set_yticks([])
    
    for idx, img_name in enumerate(random.sample(os.listdir(actual_path), num_images)):
        ax_arr[idx, 0].imshow(plt.imread(overlay_path+'/'+img_name))
        ax_arr[idx, 0].set_ylabel(img_name)
        ax_arr[idx, 1].imshow(plt.imread(inference_images_path+'/'+img_name))
        ax_arr[idx, 2].imshow(plt.imread(inference_path+'/'+img_name), cmap='gray')
        ax_arr[idx, 3].imshow(plt.imread(actual_path+'/'+img_name), cmap='gray')
    fig.tight_layout()
# visualizing random images
NUM_IMAGES = 4

visualize_images(NUM_IMAGES)
!sleep 45
!curl -v triton:8000/v2/health/ready
!curl -v triton:8000/v2/models/flood_segmentation_model
# import dependencies
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
import random

warnings.filterwarnings("ignore")
def preprocess_image(input_image): 
    image_ary=np.asarray(input_image)
    image_ary=image_ary.astype(np.float32)
    
    image_ary=(image_ary-127.5)*0.00784313725490196
    
    BGR=np.empty_like(image_ary)
    BGR[:, :, 0]=image_ary[:, :, 2]
    BGR[:, :, 1]=image_ary[:, :, 1]
    BGR[:, :, 2]=image_ary[:, :, 0]
    image_ary=BGR
    
    image_ary=np.transpose(image_ary, [2, 0, 1])
    image_ary=np.expand_dims(image_ary, axis=0)
    return image_ary
# choose random image
random_image_file=random.sample(os.listdir(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images')), 1)[0]

# preprocess
image=Image.open(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images', random_image_file))
mask=Image.open(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'masks', 'all_masks', random_image_file))
image_ary=preprocess_image(image)

print('The input array has a shape of {}.'.format(image_ary.shape)
import tritonclient.http as tritonhttpclient
from pprint import pprint

# set parameters
VERBOSE=False
input_name='input_1'
input_shape=(1, 3, 512, 512)
input_dtype='FP32'
output_name='softmax_1'
model_name='flood_segmentation_model'
url='triton:8000'
model_version='1'
# instantiate Triton Inference Server client
triton_client=tritonhttpclient.InferenceServerClient(url=url, verbose=VERBOSE)

# get model metadata
print('----------Metadata----------')
model_metadata=triton_client.get_model_metadata(model_name=model_name, model_version=model_version)
pprint(model_metadata)

# get model configuration
print('----------Configuration----------')
model_config=triton_client.get_model_config(model_name=model_name, model_version=model_version)
print(model_config)
inference_input=tritonhttpclient.InferInput(input_name, input_shape, input_dtype)
output=tritonhttpclient.InferRequestedOutput(output_name)

inference_input.set_data_from_numpy(image_ary)

# time the process
start=time.time()
response=triton_client.infer(model_name, 
                             model_version=model_version, 
                             inputs=[inference_input], 
                             outputs=[output])
latency=time.time()-start
logits=response.as_numpy(output_name)

print('The output array has a shape of {}.'.format(logits.shape))
print('It took {} per inference.'.format(round(latency, 3)))
# visualize results
fig, ax_arr=plt.subplots(1, 3, figsize=[15, 5], sharex=True, sharey=True)
ax_arr[0].set_title('Input Data')
ax_arr[1].set_title('Inference')
ax_arr[2].set_title('Actual')
ax_arr[0].set_xticks([])
ax_arr[0].set_yticks([])

ax_arr[0].imshow(image)
ax_arr[1].imshow(np.argmax(logits, axis=3)[0]*255, cmap='gray')
ax_arr[2].imshow(mask, cmap='gray')

fig.tight_layout()
plt.show()
 # DO NOT CHANGE THIS CELL
time_list=[]

for image_path in os.listdir(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images')): 
    image=Image.open(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images', image_path))
    image_ary=preprocess_image(image)
    inference_input.set_data_from_numpy(image_ary)
    
    # time the process
    start=time.time()
    response=triton_client.infer(model_name, 
                                 model_version=model_version, 
                                 inputs=[inference_input], 
                                 outputs=[output])
    time_list.append(time.time()-start)
    logits=response.as_numpy(output_name)
    
latency=sum(time_list)/len(time_list)
print('It took {} seconds to infer {} images.'.format(round(sum(time_list), 3), len(time_list)))
print('On average it took {} seconds per inference.'.format(round(latency, 3)))
batch_size=8
# create directory for model
!mkdir -p models/flood_segmentation_model_batch/1

# copy sample_resnet18.engine to the model repository
!cp $LOCAL_EXPERIMENT_DIR/export/sample_resnet18.engine models/flood_segmentation_model_batch/1/model.plan
# DO NOT CHANGE THIS CELL
configuration = """
name: "flood_segmentation_model_batch"
platform: "tensorrt_plan"
max_batch_size: {}
input: [
 {{
    name: "input_1"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 512, 512 ]
 }}
]
output: {{
    name: "softmax_1"
    data_type: TYPE_FP32
    dims: [ 512, 512, 2]
 }}
""".format(batch_size)

with open('models/flood_segmentation_model_batch/config.pbtxt', 'w') as file:
    file.write(configuration)
# DO NOT CHANGE THIS CELL
# define new input shape
batch_input_shape=(batch_size, 3, 512, 512)

batch_inference_input=tritonhttpclient.InferInput(name='input_1', shape=batch_input_shape, datatype='FP32')
batch_output=tritonhttpclient.InferRequestedOutput('softmax_1')

# create empty array for the batch input
batch_ary=np.empty(batch_input_shape).astype(np.float32)

time_list=[]

# iterate through all images
for idx, image_path in enumerate(os.listdir(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images'))): 
    image=Image.open(os.path.join(os.getenv('LOCAL_DATA_DIR'), 'images', 'all_images', image_path))
    batch_ary[idx%batch_size]=preprocess_image(image)
    if idx%batch_size==(batch_size-1): 
        batch_inference_input.set_data_from_numpy(batch_ary)

        # time the process
        start=time.time()
        response=triton_client.infer(model_name='flood_segmentation_model_batch', 
                                     model_version='1', 
                                     inputs=[batch_inference_input], 
                                     outputs=[batch_output])
        time_list.append(time.time()-start)
        logits=response.as_numpy(output_name)

batch_latency=sum(time_list)/len(time_list)
print('It took {} seconds to infer {} images.'.format(round(sum(time_list), 3), len(time_list)*batch_size))
print('On average it took {} seconds per inference.'.format(round(batch_latency, 3)))
# DO NOT CHANGE THIS CELL
# plot throughput vs. latency
plt.title('Inference Performance Comparison')
plt.plot([latency, batch_latency], [1/latency, batch_size/batch_latency], marker='o')
plt.text(latency, (1/latency)-2.5, 'Non-Batching')
plt.text(batch_latency, (batch_size/batch_latency)-2.5, 'Batching ({})'.format(batch_size))

plt.xlabel('Latency (Second)')
plt.ylabel('Throughput (Image/Second)')
plt.xlim(xmin=0, xmax=max(batch_latency, latency)*1.25)
plt.ylim(ymin=0, ymax=max(1/latency, batch_size/batch_latency)*1.25)
plt.show()
