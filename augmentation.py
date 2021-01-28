import cv2 
import numpy as np
import matplotlib.pyplot as plt


def draw_rect(im, cords, color = None):
    """Draw the rectangle on the image
    
    Parameters
    ----------
    
    im : numpy.ndarray
        numpy image 
    
    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    Returns
    -------
    
    numpy.ndarray
        numpy image with bounding boxes drawn on it
        
    """
    
    im = im.copy()
    
    cords = cords[:,:]
#     cords = cords.reshape(-1,8)
    if not color:
        color = [255,255,255]
    for cord in cords:
        
        pt1, pt2 ,pt3 ,pt4 = (cord[0], cord[1]) , (cord[2], cord[3]) ,(cord[4],cord[5]),(cord[6],cord[7])
                
#         pt1 = int(pt1[0]), int(pt1[1])
#         pt2 = int(pt2[0]), int(pt2[1])
        cnt = [pt1,pt3,pt2,pt4]
        cnt = np.asarray(cnt)
        cnt = np.int32([cnt])
        im = cv2.polylines(im, cnt,True,(0,0,0))
#         im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2])/200))
    return im

def bbox_area(bbox):
    return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])
        
def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    
    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
    
    delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    mask = (delta_area < (1 - alpha)).astype(int)
    
    bbox = bbox[mask == 1,:]


    return bbox


def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding
    
    Parameters
    ----------
    
    img : numpy.ndarray
        Image 
    
    inp_dim: tuple(int)
        shape of the reszied image
        
    Returns
    -------
    
    numpy.ndarray:
        Resized image
    
    '''

    inp_dim = (inp_dim, inp_dim)
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h))
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

import random
from tqdm import tqdm
import os
from shutil import copyfile
import cv2
import base64
import numpy as np
from tqdm import tqdm
import glob
import json
import base64

# import pickle as pkl
# from data_aug.data_aug import *
# from data_aug.bbox_util import *
import random


def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated
# Horizontal Flip:-

def horizontalFlip(img,bboxes):
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center,img_center,img_center))
    
    img = img[:, ::-1, :]
    bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])
    bboxes[:, [4, 6]] += 2*(img_center[[2, 0]] - bboxes[:, [4, 6]])
    box_w = abs(bboxes[:, 0] - bboxes[:, 2])
    box_w_1 = abs(bboxes[:, 6] - bboxes[:, 4])
    bboxes[:, 0] -= box_w
    bboxes[:, 2] += box_w
    bboxes[:, 6] -= box_w_1
    bboxes[:, 4] += box_w_1
    
    return img, bboxes


# Translate :-

def Translate(img,bboxes,translate,diff):
    
    if type(translate) == tuple:
        assert len(translate) == 2, "Invalid range"  
        assert translate[0] > 0 & translate[0] < 1
        assert translate[1] > 0 & translate[1] < 1
    else:
        assert translate > 0 and translate < 1
        translate = (-translate, translate)
        

    img_shape = img.shape
    translate_factor_x = random.uniform(*translate)
    translate_factor_y = random.uniform(*translate)

    if not diff:
        translate_factor_y = translate_factor_x
        
    corner_x = int(translate_factor_x*img.shape[1])
    corner_y = int(translate_factor_y*img.shape[0])
    canvas_r = np.random.rand(img.shape[0]+abs(corner_y),img.shape[1]+abs(corner_x))
    canvas_g = np.random.rand(img.shape[0]+abs(corner_y),img.shape[1]+abs(corner_x))
    canvas_b = np.random.rand(img.shape[0]+abs(corner_y),img.shape[1]+abs(corner_x))
    canvas = np.stack((canvas_r, canvas_g, canvas_b), axis=-1)
    canvas *= 255.0
    canvas = canvas.astype(np.uint8)

    orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
    
    random_noise =random.randint(0,20)
    
    if corner_x > 0 and corner_y > 0:
        canvas[orig_box_cords[0]:canvas.shape[0], orig_box_cords[1]:canvas.shape[1],:] = img
        bboxes[:,:] += [corner_x, corner_y, corner_x, corner_y,corner_x, corner_y,corner_x, corner_y]

    elif corner_x <0 and corner_y > 0:
        canvas[orig_box_cords[0]:canvas.shape[0], orig_box_cords[1]:canvas.shape[1]+corner_x,:] = img
        bboxes[:,:] += [0, corner_y, 0, corner_y,0, corner_y, 0, corner_y]

    elif corner_x >0 and corner_y < 0:
        canvas[orig_box_cords[0]:canvas.shape[0]+corner_y, orig_box_cords[1]:canvas.shape[1],:] = img
        bboxes[:,:] += [corner_x, 0, corner_x, 0,corner_x, 0, corner_x, 0]

    elif corner_x <0 and corner_y <0:
        canvas[orig_box_cords[0]:canvas.shape[0]+corner_y, orig_box_cords[1]:canvas.shape[1]+corner_x,:] = img
    
    img = canvas

    return img, bboxes
    
# Rotation :-

def Rotation(img,bboxes,angle):
    
    if type(angle) == tuple:
        assert len(angle) == 2, "Invalid range"  
    else:
        angle = (-angle, angle)

    angle = random.uniform(*angle)
    
    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)

    corners = bboxes

    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)

    new_bbox = corners

    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))

    new_bbox[:,:] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

    bboxes  = new_bbox
#     bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

    return img, bboxes

def shear(img,bboxes,shear_factor):
    if type(shear_factor) == tuple:
            assert len(shear_factor) == 2, "Invalid range for scaling factor"   
    else:
        shear_factor = (-shear_factor, shear_factor)
    shear_factor = random.uniform(*shear_factor)
    
    w,h = img.shape[1], img.shape[0]

    if shear_factor < 0:
        img, bboxes = horizontalFlip(img, bboxes)

    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

    nW =  img.shape[1] + abs(shear_factor*img.shape[0])

    bboxes[:,[0,2,4,6]] += ((bboxes[:,[1,3,5,7]]) * abs(shear_factor) ).astype(int) 

    img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

    if shear_factor < 0:
        img, bboxes = horizontalFlip(img, bboxes)

    img = cv2.resize(img, (w,h))

    scale_factor_x = nW / w
    
    bboxes[:,:] /= [[scale_factor_x, 1, scale_factor_x, 
                     1,scale_factor_x, 1, scale_factor_x, 
                     1] for b in range(len(bboxes)) ]


    return img, bboxes
import cv2
import numpy as np
import base64
def base64toimg(encoded_data):
    nparr = np.frombuffer(encoded_data, dtype=np.uint8)
    img = cv2.imdecode(nparr, flags=1)
    return img
# -----------------------------------Main Function Starting-------------------------------------------------------
for countss in range(0,2):
    for i in tqdm(glob.glob('data_annotated_train/*.json')):
        try:
            with open(i,'r') as f:
                dicts = json.loads(f.read())
            imgdata = base64.b64decode(dicts['imageData'])
            img = base64toimg(imgdata)
#             print(img)
#             cv2.imwrite('')
#             img = cv2.imread(i.replace('.json','.jpg'))
#             if True/:#'election_old_front' == dicts['shapes'][0]['label']:
            bboxes = []
            for shape_index in range(len(dicts['shapes'])):
                tmp_list = []
                for tmp_k in range(0,4):
                    tmp_list.append(sum(dicts['shapes'][shape_index]['points'][tmp_k]))
                min_index = tmp_list.index(min(tmp_list))
                max_index = tmp_list.index(max(tmp_list))
                x_min = dicts['shapes'][shape_index]['points'][min_index][0]
                y_min = dicts['shapes'][shape_index]['points'][min_index][1]
                x_max = dicts['shapes'][shape_index]['points'][max_index][0]
                y_max = dicts['shapes'][shape_index]['points'][max_index][1]
                tmp_list = []
                for m in range(0,4):
                    if m != min_index and m != max_index:
                        tmp_list.append(m)
                if dicts['shapes'][shape_index]['points'][tmp_list[0]][0] < dicts['shapes'][shape_index]['points'][tmp_list[1]][0]:
                    x_max_left = dicts['shapes'][shape_index]['points'][tmp_list[1]][0]
                    y_max_left = dicts['shapes'][shape_index]['points'][tmp_list[1]][1]
                    x_min_right = dicts['shapes'][shape_index]['points'][tmp_list[0]][0]
                    y_min_right = dicts['shapes'][shape_index]['points'][tmp_list[0]][1]
                else:
                    x_max_left = dicts['shapes'][shape_index]['points'][tmp_list[0]][0]
                    y_max_left = dicts['shapes'][shape_index]['points'][tmp_list[0]][1]
                    x_min_right = dicts['shapes'][shape_index]['points'][tmp_list[1]][0]
                    y_min_right = dicts['shapes'][shape_index]['points'][tmp_list[1]][1]


                bboxes.append([x_min,y_min,x_max,y_max,x_max_left,y_max_left,x_min_right,y_min_right])
            bboxes=np.asarray(bboxes)

            Total_iteration=[1,2,3,4]
            Total_iteration=random.choice(Total_iteration)
    #         print('Total_iteration is:->',Total_iteration)

            img_ = img
            bboxes_ = bboxes
            Count = [1,2,3,4]
            for row in range(Total_iteration):

                agumentation_list=['horizontalFlip','Shear','Rotation','Translate']
                choice=random.choice(agumentation_list)

                if(choice=='horizontalFlip'):
    #                 print(bboxes_,':::::::::::::::')
                    img_,bboxes_ = horizontalFlip(img_,bboxes_)
        #             print('Horizontal Flip is Completed')
        #             plotted_img = draw_rect(img_, bboxes_)
        #             plt.imshow(plotted_img)
        #             plt.show()
        #         elif(choice=='Rotation'):
        #             img_, bboxes_ = Rotation(img_,bboxes_,20)
        #             plotted_img = draw_rect(img_, bboxes_)
        #             print('Rotation is Completed')
        #             plt.imshow(plotted_img)
        #             plt.show()


        #         elif(choice=='Translate'):
        #             Count=[1,2,3,4]
        #             Total_Number_Translate=random.choice(Count)
        #             print('Total_Number_Translate',Total_Number_Translate)
        #             for row in range (Total_Number_Translate):
        #                 img_, bboxes_=Translate(img_,bboxes_,0.3,True)
        #                 plotted_img = draw_rect(img_, bboxes_)
        #                 print('Translate is Completed')
        #                 plt.imshow(plotted_img)
        #                 plt.show()
                elif(choice=='Shear'):
                    img_,bboxes_ = shear(img_,bboxes_,0.3)

        #                 print('Shear Flip is Completed')
        #                 plotted_img = draw_rect(img_, bboxes_)
        #                 plt.imshow(plotted_img)
        #                 plt.show()

                elif(choice=='Rotation'):
                    img_, bboxes_ = Rotation(img_,bboxes_,5)
    #                 plotted_img = draw_rect(img_, bboxes_)
        #                 print('Rotation is Completed')
        #                 plt.imshow(plotted_img)
        #                 plt.show()


                elif(choice=='Translate'):
                    if Total_iteration < 2:
                        Total_Number_Translate=random.choice(Count)
                        for row in range (Total_Number_Translate):
                            img_, bboxes_=Translate(img_,bboxes_,0.1,True)
                    else:
                        Total_Number_Translate = 1
            #                 print('Total_Number_Translate',Total_Number_Translate)
                        for row in range (Total_Number_Translate):
                            img_, bboxes_=Translate(img_,bboxes_,0.3,True)
    #         print(len(bboxes_),':>>>>>>>>>>>>>>>>>>>>>>>>>')
            for k in range(len(bboxes_)):
                x_min = bboxes_[k][0]
                y_min = bboxes_[k][1]
                x_max = bboxes_[k][2]
                y_max = bboxes_[k][3]
                x_max_left =   bboxes_[k][4]
                y_max_left =   bboxes_[k][5]
                x_min_right =   bboxes_[k][6]
                y_min_right =   bboxes_[k][7]

                # Updating Current Cooridinate Value into Dictionary:-
                dicts['shapes'][k]['points'][0][0] = x_min
                dicts['shapes'][k]['points'][0][1] = y_min
                dicts['shapes'][k]['points'][2][0] = x_max
                dicts['shapes'][k]['points'][2][1] = y_max
                dicts['shapes'][k]['points'][1][0] = x_max_left
                dicts['shapes'][k]['points'][1][1] = y_max_left
                dicts['shapes'][k]['points'][3][0] = x_min_right
                dicts['shapes'][k]['points'][3][1] = y_min_right

            # Storing Image:-
            output=i
            output=output.split('/')
            last_one=output[-1]
            output_file_name=last_one.replace('.json','')
            cv2.imwrite('output/'+ output_file_name+'_'+str(countss)+'.jpg',img_)

            # Target image Size:-
            targetSize_x=img_.shape[0]
            targetSize_y=img_.shape[1]

            # Update Heihgt and Weight into Dictionary:-
            dicts['imageHeight'] = targetSize_x
            dicts['imageWidth'] = targetSize_y

            # Stroing JSON File:-
        #     object_name_agument_number=
            with open('output/'+ output_file_name+'_'+str(countss)+'.jpg', "rb") as image_file: 
                my_string = base64.b64encode(image_file.read())
            dicts['imageData'] = my_string.decode('utf-8')
            dicts['imagePath'] = output_file_name+'_'+str(countss)+'.jpg'
            # Write Updated Dictionary to the JSON File:-
            output_file_name=last_one.replace('.json','')
            with open('output/'+output_file_name+'_'+str(countss)+'.json','w') as f:
                json.dump(dicts,f)
#                 print('...................')
        except Exception as e:
            print(e)
