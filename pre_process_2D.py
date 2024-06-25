import os
import cv2
import numpy as np
from tqdm import tqdm
import cc3d
from multiprocessing import Pool
import concurrent.futures
import json
import SimpleITK as sitk
from PIL import Image
join = os.path.join
voxel_num_thre2d = 100
voxel_num_thre3d = 1000
sum2=0
sum3=0
output_folder = "/notebooks/datasets/png/CT/AVT"   
img_path="/notebooks/datasets/AVT/imgs"
mask_path="/notebooks/datasets/AVT/gts"
sumw=0
sumx=0
sumy=0
s1=0
s2=0


def write_category_info_to_json(category_info, json_file_path):
    """
    Write category_info dictionary to a JSON file, merge lists if keys already exist.

    Parameters:
    - category_info: Dictionary containing category information.
    - json_file_path: Path to the JSON file.

    Returns:
    - None
    """
    global sumw,sumx,sumy,s1,s2
    sumw+=1
    
    try:
        with open(json_file_path, 'r') as json_file:
            existing_data = json.load(json_file)
    except FileNotFoundError:
        existing_data = {}
    sumx+=1
    for key, value in category_info.items():
        if key in existing_data:
            # Key already exists, merge lists
            try:
                existing_data[key].extend(value)
            except FileNotFoundError:
                print(key,value)
            s1+=1
        else:
            # Key does not exist, add new key-value pair
            existing_data[key] = value
            s2+=1
    # Save the updated dictionary as a JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(existing_data, json_file)
    sumy+=1


def normalize_min_max(image):
    """
    Perform self-min-max normalization on the input image to bring pixel values to [0, 1].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val- min_val)
    return normalized_image


def split_labels_binary_foregrounds(mask_data):
    """
    Split original labels into binary foregrounds.

    Parameters:
    - mask_data: The original label mask image.

    Returns:
    - A stack of binary foregrounds corresponding to each non-zero label.
    """
    # print(mask_data.shape)
    unique_labels = np.unique(mask_data)
    # print(unique_labels)
    if(len(unique_labels)==1):
        return []
    # Exclude zero label (background) and extract non-zero labels as foregrounds
    foregrounds = [np.uint8(mask_data == label)*255 for label in unique_labels if label != 0]
    # print(np.unique(foregrounds))
    return np.stack(foregrounds, axis=0)

def separate_foreground(mask):
    # mask[mask == 255] = 1
    # print(np.unique(mask))
    # 查找连接组件
    min_area_threshold=100
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    unique_masks = []

    for i in range(1, num_labels):
        # 为每个连通域生成唯一的标签
        # 忽略小于阈值的区域
        if stats[i, cv2.CC_STAT_AREA] < min_area_threshold:
            continue
        unique_mask = np.zeros_like(mask, dtype=np.uint8)
        # print(s,unique_mask.shape,labels.shape)
        unique_mask[labels == i] = 1
        unique_masks.append(unique_mask)

    return unique_masks#, unique_labels



def separate_connected_components_and_consolidate(mask_data, threshold_size):
    """
    Separate foregrounds into distinct connected components and consolidate smaller segmentation targets.
    """
    # print(mask_data.shape)
    # Separate connected components
    separated_components, num_components = cc3d.connected_components(mask_data, connectivity=26, return_N=True)

    # Consolidate smaller segmentation targets
    consolidated_mask = cc3d.dust(separated_components, threshold=threshold_size, connectivity=26, in_place=True)
    # 将所有为 1 的地方修改为 255
    consolidated_mask[consolidated_mask == 1] = 255
    return consolidated_mask

def filter_masks_by_size(masks, min_size_percentage):
    """
    Filter out masks if the target area accounts for less than a percentage of the total image area.
    """
    masks[masks == 1] = 255
    # print(np.unique(masks))
    _,a,b,c= masks.shape
    # if(a==1):
    #     f1=2
    #     f2=3
    # if(b==1):
    #     f1=1
    #     f2=3
    # if(c==1):
    #     f1=1
    #     f2=2
    total_area = masks.shape[2] * masks.shape[3]
    mask_areas = np.sum(masks//255, axis=(2, 3))
    
    # print(np.unique(masks),np.unique(masks// 255))
    valid_masks = masks[mask_areas >= min_size_percentage * total_area]
    # print(mask_areas,min_size_percentage * total_area)

    # print(len(valid_masks))
    return valid_masks



def save_masks_in_png_format(masks, output_folder, naming_convention):
    """
    Save validated masks in PNG format.
    """
    for i, mask in enumerate(masks):
        instance_id = naming_convention["instance_ids"][i]
        mask_path = os.path.join(output_folder, f"{instance_id}.png")
        cv2.imwrite(mask_path, mask)




def unify_data_dimension(file_name):
    # Check if the file is an image (you might want to add more file format checks)
    if file_name.endswith(".nrrd"):
        # Assuming you have a function to load 3D image data, replace the following line
        # with the appropriate code to load the 3D image data
        # load image and preprocess
        
        img_sitk = sitk.ReadImage(join(img_path, file_name))
        
        image_data = sitk.GetArrayFromImage(img_sitk)
        
        mask_sitk = sitk.ReadImage(join(mask_path, file_name.split('.')[0]+'.seg.nrrd'))  #&&&&&&&&&&&&&&&&&
        mask_data = np.uint8(sitk.GetArrayFromImage(mask_sitk))
        # image_data = load_3d_image(os.path.join(input_folder, file_name))
        # Load mask data (replace with the appropriate code to load mask data)
        # mask_data = load_mask_data(input_folder)
        
        # Check if the image is 3D or 2D
        if len(image_data.shape) == 2:
            # print(image_data.shape)
            # Calculate aspect ratio
            max_v, min_v = sorted(image_data.shape, reverse=True)[:2]
            # aspect_ratio = min_v / max_v
            # Discard images with extreme aspect ratios
            if min_v >= 0.5*max_v:
                # print(slice_image.shape)
                # Normalize pixel values to [0, 1]
                normalized_slice = normalize_min_max(image_data)
                
                # Take the ceiling value after multiplying by 255
                image_uint8 = np.ceil(normalized_slice * 255.0).squeeze().astype(np.uint8)                     
                # Save the processed image in PNG format
                output_file_name = file_name
                # print(os.path.join(output_folder,"images/"+ output_file_name))
                # print(np.min(slice_image_uint8), np.max(slice_image_uint8))
                
                # print(save_path,slice_image_uint8.shape)
                
                # Step 1: Split original labels into binary foregrounds
                
                # print(output_file_name)
                foregrounds = split_labels_binary_foregrounds(mask_data)
                if(len(foregrounds)==0):#跳过全黑的
                    # print("1111")
                    return
                
                # Step 2: Separate connected components and consolidate smaller segmentation targets
                consolidated_masks = []
                # for foreground in foregrounds:
                #     separated_and_consolidated = separate_connected_components_and_consolidate(foreground, threshold_size=voxel_num_thre2d)
                #     consolidated_masks.append(separated_and_consolidated)
                # print(separated_and_consolidated.shape)
                # print(len(foregrounds))
                for foreground in foregrounds:
                    # print(slice_image_uint8.shape,foreground.shape)
                    separated_and_consolidated = separate_foreground(foreground.squeeze())
                    for t in separated_and_consolidated:
                        consolidated_masks.append(np.expand_dims(t, axis=0))
                # print(consolidated_masks[0].shape)
                global sum2,sum3
                sum2=sum2+len(consolidated_masks)
                
                # print(np.unique(np.stack(consolidated_masks)))
                # Step 3: Filter masks by size
                valid_masks = filter_masks_by_size(np.stack(consolidated_masks), min_size_percentage)
                
                if(len(valid_masks)==0):#跳过被筛选完的
                    # print("3333")
                    return
                sum3=sum3+len(valid_masks)
                # print(len(valid_masks))
                # print(np.unique(valid_masks))
                # print("4444")
                cv2.imwrite(os.path.join(output_folder+"/images",output_file_name), image_uint8)
                
                # Save validated masks in PNG format
                for i, m in enumerate(valid_masks):
                    # print(m.shape,np.count_nonzero(m == 255))
                    if(np.any(m)):
                        # print(m.squeeze())
                        cv2.imwrite(join(output_folder, "masks/"+output_file_name.split('.')[0]+'_'+str(i)+'.png'), m.astype(np.uint8))
                        # Create JSON file with category information
                        category_info = {"images/"+output_file_name: ["masks/"+output_file_name.split('.')[0]+'_'+str(i)+'.png']}  # Replace with actual category information
                        write_category_info_to_json(category_info, os.path.join(output_folder, "SAMed2D_v1.json"))

                


        
# # # 读取PNG图像
# # image_path = "/notebooks/datasets/png/CT/MSD-Colon/masks/colon_005_X_52_0.png"
# image_path = "/notebooks/datasets/SAMed2Dv1/masks/ct_00--Totalsegmentator_dataset--s0065--y_0051--0047_000.png"
# image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
# count_255 = np.count_nonzero(image == 255)

# print(f"值为 255 的元素个数：{count_255}")
# print(np.unique(image))

os.makedirs(join(output_folder, "images"), exist_ok=True)
os.makedirs(join(output_folder, "masks"), exist_ok=True)
#建立一个空的json文件
empty_data = {}
# Save the empty dictionary as a JSON file
with open(os.path.join(output_folder, "SAMed2D_v1.json"), 'w') as json_file:
    json.dump(empty_data, json_file)
min_size_percentage = 0.00153  # Equivalent to 0.153% (as described in the text)
# Ensure output folder exists
# List all files in the input folder
file_list = os.listdir(img_path)
# Specify the number of threads
num_threads = 1 # Adjust this based on your system and available resources
# Use ThreadPoolExecutor for multi-threading
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit the processing tasks for each image in parallel
    futures = [executor.submit(unify_data_dimension,file_name) for file_name in file_list]

    # Use tqdm to display progress
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        pass  # Wait for all tasks to complete
print(sum2,sum3,sumw,sumx,sumy,s1,s2)
