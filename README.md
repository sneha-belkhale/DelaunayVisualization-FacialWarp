# DelaunayVisualization-FacialWarp
This is a python script that takes in a portrait image, uses built in OpenCV Haar Cascades to find the 68 facial landmarks, 
and displays a cool visualization of its delauney subdivision.  

![alt tag](http://www.freedigitalphotos.net/images/img/homepage/golf-1-top-82328.jpg)

Then (inspired by the snapchat filters) I use the delaunay triangulation and affine warping to transform the face into a.. 
chubby baby..? 

I plan on experimenting with this facial warping more so that this filtering can be done autonomously 
(right now it is slightly catered to the michaelcera.jpg image provided)
