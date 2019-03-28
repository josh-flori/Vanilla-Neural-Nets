# IMAGE RESIZING

from PIL import Image
import os
i=1
for filename in sorted([filename for filename in os.listdir('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/images original')]):
    if 'jpg' in filename:
            image=Image.open('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/images original/'+str(filename)).convert('L')
            image.thumbnail((100,60), Image.ANTIALIAS)
            image.save('/users/josh.flori/drive_backup/drive_backup/pychrm_networks/data/resized_larger_black_white/' + str(i)+'.jpg', "JPEG")
            i+=1