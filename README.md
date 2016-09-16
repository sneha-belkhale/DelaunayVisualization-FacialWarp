# DelaunayVisualization-FacialWarp
This is a python script that takes in a portrait image, uses built in OpenCV Haar Cascades to find the 68 facial landmarks, 
and displays a cool visualization of its delauney subdivision.  
![Alt text](/michaelcera.jpg "Original Michael Cera")
b&w:
![Alt text](/Screen Shot 2016-09-13 at 11.08.06 PM.png?raw=true "Delaunay Visualization")
colorized:
![Alt text](/Screen Shot 2016-09-16 at 2.53.50 PM.png?raw=true "Delaunay Visualization")
![Alt text](/Screen Shot 2016-09-16 at 2.53.41 PM.png?raw=true "Delaunay Visualization")


Then (inspired by the snapchat filters) I use the delaunay triangulation and affine warping to transform the face into a.. 
chubby baby..

![Alt text](/Screen Shot 2016-09-13 at 11.19.29 AM.png?raw=true "Face Warping")

I plan on experimenting with this facial warping more so that this filtering can be done autonomously 
(right now it is slightly catered to the michaelcera.jpg image provided)
