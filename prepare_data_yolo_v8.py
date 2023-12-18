from opts import parse_opts_offline
import os
import xml.etree.ElementTree as ET
import shutil
from sklearn.model_selection import train_test_split
import yaml
# extract data from xml file

def extract_data_from_xml(opts):
    annot_folder = os.path.join(opts.root_path, opts.annot_files)
    img_paths = [] 
    img_sizes = [] 
    img_labels = [] 
    classes = []
    class_to_idx = {}
    
    annot_files = os.listdir(annot_folder)
    for file in annot_files:
        xml_path = os.path.join(annot_folder,file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_name = root[1].text
        width,height = int(root[2][0].text),int(root[2][1].text)
        labels = []
        for obj in root.findall("object"):
            class_name = obj[0].text
            if class_name not in classes:
                classes.append(class_name)
                class_to_idx[class_name] = len(classes) - 1
            class_idx = class_to_idx[class_name]
            bb = obj[-1]
            x_min,y_min,x_max,y_max = int(bb[0].text),int(bb[1].text),int(bb[2].text),int(bb[3].text)
            
            labels.append([class_idx,x_min,y_min,x_max,y_max])
        img_paths.append(img_name)
        img_sizes.append((width,height))
        img_labels.append(labels)
    return img_paths,img_sizes,img_labels,classes,class_to_idx

                
        
    
   
def convert_to_yolo_v8_format(image_paths , image_sizes , image_labels):
    yolov8_data = []
    for image_path , image_size , labels in zip(image_paths , image_sizes , image_labels):
        image_width , image_height = image_size
        yolov8_labels = []
        for bbox in labels:
            class_idx,x_min,y_min,x_max,y_max = bbox
            w = x_max - x_min
            h = y_max - y_min
            center_x = (x_min+w/2)/image_width
            center_y = (y_min+h/2)/image_height
            width = w/image_width
            height = h/image_height
            
            yolov8_label = f"{class_idx} {center_x} {center_y} {width} {height}"
            yolov8_labels.append(yolov8_label)
        yolov8_data.append((image_path,yolov8_labels))
    return yolov8_data

def save_data(data,opts,data_type):
    save_dir = os.path.join(opts.yolo_data_dir,data_type)
    # create folder if it not exits
    os.makedirs(save_dir,exist_ok=True)
    
    # make images and labels folder
    os.makedirs(os.path.join(save_dir,"images"),exist_ok=True)
    os.makedirs(os.path.join(save_dir,"labels"),exist_ok=True)
    
    for image_path,labels in data:
        # copy image
        shutil.copy(
            os.path.join(opts.root_path,"images",image_path),
            os.path.join(save_dir,'images'),
        )
        
        # Save labels to labels folder
        image_name = os.path.basename(image_path) # example: if url = a/b => return b 
        image_name = os.path.splitext(image_name)[0] # this function return root and ext part. And then I get root part
        with open(os.path.join(save_dir,"labels",f"{image_name}.txt"),'w') as f:
            for label in labels:
                f.write(f'{label}\n')
        
def split_train_test_val(data,seed = 0,val_size = 0.1,test_size = 0.1,is_shuffle = True):
    train_data , val_data = train_test_split(
        data , test_size=val_size , random_state=seed, shuffle=is_shuffle
        )
    train_data , test_data = train_test_split(
        train_data , test_size=test_size , random_state=seed, shuffle=is_shuffle
        )
    print("train: ",len(train_data))
    print("val: ",len(val_data))
    print("test: ",len(test_data))
    return train_data,test_data,val_data

def create_yaml_file(opts,class_labels,nc = 1):
    data_yaml = {
        "path": os.path.join(opts.yolo_data_yaml,opts.yolo_data_dir),
        "train":"train/images",
        "val":"val/images",
        "test":"test/images",
        "nc" : nc,
        "names":class_labels
    }
    yolo_yaml_path = os.path.join(opts.yolo_data_dir,'data.yml')
    with open(yolo_yaml_path,'w') as f:
        yaml.dump(data_yaml,f,default_flow_style=False)
    
    

if __name__ == '__main__': 
    opts = parse_opts_offline()
    img_paths,img_sizes,img_labels,classes,class_to_idx = extract_data_from_xml(opts)
    yolov8_data = convert_to_yolo_v8_format(img_paths , img_sizes,img_labels)
    train_data,test_data,val_data = split_train_test_val(yolov8_data)
   
    # save data according to yolov8 format
    save_data(train_data,opts,"train")
    save_data(test_data,opts,"test")
    save_data(val_data,opts,"val")
    
    create_yaml_file(opts,class_labels=classes,nc = len(classes))
    
    