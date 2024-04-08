import cv2
import numpy as np

width = 3840
height = 2160

scale_x = (1280) / width
scale_y = (720) / height

#f_name = "../../Labels/MultiActionLabels/3840x2160/1.1.1.txt"
f_name = "../../Labels/SingleActionTrackingLabels/3840x2160/1.1.1.txt"
lines = []
with open(f_name,"r") as f:
    lines = f.readlines()

orig_frames = [230 + i for i in range(306)]

bboxes = {}
people = set()
same_people = {}
for line in lines:
    s = line.split(" ")
    frame = int(s[5])
    if frame not in bboxes:
        bboxes[frame] = set()
    new_coord = (int(s[1]),int(s[2]),int(s[3]),int(s[4]),int(s[0]))
    curr_person = int(s[0])
    if curr_person not in same_people:
        same_people[curr_person] = set()
    xc = (new_coord[0] + new_coord[2]) / 2
    yc = (new_coord[1] + new_coord[3]) / 2
    if len(bboxes[frame]) == 0:
        bboxes[frame].add(new_coord)
    else:
        add_bbox = True
        for (x1,y1,x2,y2,t) in bboxes[frame]:
            cxc = (x1 + x2) / 2
            cyc = (y1 + y2) / 2
            dist = np.sqrt((xc - cxc)**2 + (yc - cyc)**2)
            if dist < 20:
                add_bbox = False
                same_people[curr_person].add(t)
        if add_bbox:
            bboxes[frame].add(new_coord)

colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),(30,30,30),(125,0,125),(200,200,200)]
colors_to_people = {}

for frame in orig_frames:
    for (_,_,_,_,t) in bboxes[frame]:
        people.add(t)

for idx,k in enumerate(people):
    colors_to_people[k] = colors[idx]

#print(colors_to_people)
#print(len(colors_to_people))
#exit(-1)

for idx,frame in enumerate(orig_frames):
    f1 = open("labels/frame_{}.txt".format('%05d'%(idx+1)),"w")
    mask = np.zeros((720//2,1280//2,3))
    jdx = 0
    if len(bboxes[frame]) > 9:
        print(frame)
        print(idx)
        print(len(bboxes[frame]))
        print(bboxes[frame])
    bboxes_for_frame = sorted(bboxes[frame],key=lambda x: int(x[4]))
    temp = bboxes_for_frame[0]
    bboxes_for_frame[0] = bboxes_for_frame[2]
    bboxes_for_frame[2] = temp
    #new_img = cv2.imread("og_images/frame_{}.jpg".format("%05d"%(idx+1)))
    for (xmin,ymin,xmax,ymax,t) in bboxes_for_frame:
        if idx == 130 and t == 5:
            continue
        elif t == 19:
            t = 5
        start_coord = (int(xmin*scale_x),int(ymin*scale_y))
        end_coord = (int(xmax*scale_x),int(ymax*scale_y))
        #new_img = cv2.rectangle(new_img,(start_coord[0]*2,start_coord[1]*2),(end_coord[0]*2,end_coord[1]*2),colors_to_people[t],-1)
        mask = cv2.rectangle(mask,start_coord,end_coord,colors_to_people[t],-1)
        x_c = (xmin + ((xmax - xmin) / 2)) / width
        y_c = (ymin + ((ymax - ymin) / 2)) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        f1.write("0 %0.6f %0.6f %0.6f %0.6f\n" %(x_c,y_c,w,h))
        jdx += 1
    f1.close()
    cv2.imwrite("masks/mask_{}.png".format('%05d'%(idx+1)),mask)
    #cv2.imwrite("images/frame_{}.jpg".format('%05d'%(idx+1)),new_img)
