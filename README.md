### Models


Dataset|Backbone|MODEL |Download Links 
:------------: |:------------: |:------------: |:------------:
COCO|R-50|BshapeNet+(PANet)(pytorch detectron) | [model](https://drive.google.com/open?id=1ypC62vZVSGK3QGnZTXkZpDsxavF04MAn) 
COCO|X-101| BshapeNet+ Detectron| [model](https://drive.google.com/open?id=1bo9g7pymsi3-SvMYFTiM0_a_9TziJ0FB) 
CITY|R-50|BshapeNet+ Detectron| [model](https://drive.google.com/open?id=1JGIyzHu7Xf-hHAmJ07Bea7vgIvWEF1Rp) 



### MS COCO Results
#### Object detection
MODEL|Backbone|AP|AP<sup>50|AP<sup>75|AP<sup>L|AP<sup>M|AP<sup>S 
:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: 
BshapeNet+(PANet) | R-50 | 44.2	|63.5|46.7	|56.4	|46.8	|27.4
BshapeNet+|X-101|42.8| 64.9| 46.9| 53.6| 46.1| 25.2

#### Instance segmentation
MODEL|Backbone|AP|AP<sup>50|AP<sup>75|AP<sup>L|AP<sup>M|AP<sup>S 
:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: 
BshapeNet+(PANet) | R-50 | 37.4| 58.7| 40.1| 53.3| 38.7| 17.1
BshapeNet+|X-101|37.9 |61.3| 40.2| 54.4| 40.4| 17.4
  
### Cityscapes Results
MODEL|Backbone|AP|person|rider|car|truck|bus|train|motorcycle|bicycle
:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: |:-------: 
BshapeNet+[fine-only]|R-50|27.3|29.7|23.4|46.7|26.1|33.3|24.8|20.3|14.1
BshapeNet+[COCO]|R-50|32.9|36.6|24.8|50.4|33.7|41.0|33.7|25.4|17.8
