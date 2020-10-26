import torch
import numpy as np
import numpy

def _make_array(t):
    # Use float64 for higher precision, because why not?
    # Always put polygons on CPU (self.to is a no-op) since they
    # are supposed to be small tensors.
    # May need to change this assumption if GPU placement becomes useful
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    return np.asarray(t).astype("float64")
count=0
dictt=torch.load('./test_data/file.pth')
dictt2=torch.load('./test_data/file.pth')
print(' ori')
print(len(dictt2))
for item in dictt:
    for anno in item['annotations']:
        seg=anno['segmentation']
        polygons_per_instance = [_make_array(s) for s in seg]
        for polygon in polygons_per_instance:
            if(polygon is None or len(polygon)<6):
                count+=1
                try:
                    dictt2.remove(item)
                except:
                    break
print(' after')
print(len(dictt2))

print(count)
torch.save(dictt2,'./test_data/file.pth')

