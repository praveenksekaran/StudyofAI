# Stable Diffusion

Gonkee
[youtube](https://www.youtube.com/watch?v=sFztPP9qPRc)

# Intro 
stable diffusion beats Generative Adversarial Networks (GAN)

#### Links 

Kaggle - AI & ML projects, datasets [kaggle](https://www.kaggle.com/)

#### Fully conected layer 
each neuron is connected every nueron in the other layer 
![image](https://github.com/user-attachments/assets/1d71b0e9-b864-4420-8b82-afdcf8e9293e)


#### Image generation
is dependant of 2 special types of layers. each play a special role. 
![image](https://github.com/user-attachments/assets/2c421a10-264c-4976-bb04-a08bb8b8fcc8)

#### Convolution Layer 
each pixle is represented by grid of numbers called kernel it can be 3x3 or 5x5  
![image](https://github.com/user-attachments/assets/296686f1-fef2-4224-a63f-20750ca48ce7)

# Computer vision 

#### Level 1 - Image Classification
network labels it as fish. usually single object 
![image](https://github.com/user-attachments/assets/28cf8306-1682-4d97-b34f-841c3fb4a35a)

#### Level 2 - Image Classification + Localization
network labels it as fish. and identifies its location. usually single object 
![image](https://github.com/user-attachments/assets/f688951c-1ebc-4caa-a1e6-021bdcd3fbe2)

#### Level 3 - Object detection
network identifies multiple object and identifies its location. labels them  
![image](https://github.com/user-attachments/assets/ab3ba6f7-7f80-49ea-b848-e9dde963c32f)


#### Level 4 - Sementic Segmentation
each image gets labeled. exact shape is identified  
![image](https://github.com/user-attachments/assets/597f6cec-3086-426a-b84f-baa4e7be32d3)

#### Level 5 - Instance Segmentation
exact shape is identified. each object is labeled induvidually. 
![image](https://github.com/user-attachments/assets/bfba0f19-33a1-4841-a1c9-7bdab275f18f)

# U-net : convolution networks for biomedical imaging
scales down images and then scales up to original resolution.
![image](https://github.com/user-attachments/assets/7fa10200-6f8b-4f4f-913b-b8178fb9988a)

unet segment images so efficiently remember prior methods required thousands of sample images but I've only
8:45
given this one 500 images and it's doing pretty well so when this RGB image of a fish gets inputed into the unet it's
8:52
represented in computer memory as a 3D grid of numbers because it has a width
8:57
height and three channels so this is a three-dimensional tensor in machine learning language
![image](https://github.com/user-attachments/assets/0db33a44-3fbf-4fa8-a76e-858e9967f6e2)

9:04
now at the start theimage only has three channels to represent redness greenness and bless
9:09
but what if it could have more channels to represent more information like what part of the image corresponds to the
9:16
body of the fish what part is the cutting board what part is the shadow what part is the highlights and so on so
9:22
that's essentially the whole point of convolutions it's to extract features from an image from how the pixels relate
9:28
to each each other and what makes convolutions even more powerful is when there's more channels in the image than
9:34
just one channel because then the kernel is a 3D grid instead of just a 2d
![image](https://github.com/user-attachments/assets/075720fc-cc38-413a-982f-c040ff3653be)



9:41
one the first half of the unit has all these convolutional blocks that makes the number of channels in the image go 
from 3 to 64 to 128 to 256 to 512 and
9:51
finally to 1,24 in the convolution from 64 to 128 channels for example each kernel is 64
![image](https://github.com/user-attachments/assets/465919f4-9a90-4668-ab8f-570b5860bab0)


9:58
layers deep and there's 128 of those kernels that's how the network can
10:03
extract more and more complex features from the image slight issue though even though the kernels get deeper and deeper
10:10
they still have a fixed field of view on the image in this case a 3X3 field of
10:16
view in order to better extract features from the image obviously the kernels are going to have to see more of the image
10:21
so how can we make the field of view bigger well just making the kernels bigger rapidly increases the number of
10:28
parameters that we have have which makes it inefficient so the unit uses a really smart and efficient alternative if we
10:34
can't make the kernels bigger then just make the image smaller so after every
![image](https://github.com/user-attachments/assets/29278882-6e5a-43ea-88ef-c2bcc0c66309)

10:40
two convolutional blocks the image gets scaled down before it goes into the next two convolutional blocks this increased
10:46
field of view is how the network can capture more context within the image to better understand it so let's see what
10:53
our fish has turned into in the middle of the unit where there's the most number of channels but the resolution is
10:59
is the smallest out of the 1024 channels we can see that some of them highlight the body of the fish some of them the
11:05
background some of them highlight the brighter area above the fish and some of them the darker area below just as we
11:12
said before so at this point the network has learned all the information on what is in the image but the downscaling has
11:19
made it lose information on where it is in the image so in the second half of the unit we start scaling it back up
![image](https://github.com/user-attachments/assets/f5f31c01-1cca-4a97-a97a-7bc35a36bea1)

11:26
again and decrease the number of channels using the these convolutional blocks to kind of consolidate and
11:32
summarize up all that information that we gathered in the first half but how do we get back all the Lost detail from the
11:37
down sampling in the first half the answer is what's known as residual connections where every time the

![image](https://github.com/user-attachments/assets/c0bdf4ea-6815-43f7-bf3d-234e42597a38)

11:43
resolution is increased the information from the previous time the image was that resolution is literally just
11:49
slapped onto the back and combined with it and then the convolutional layers mix the information back in if we compare
11:55
the fish image at its highest resolution in the beginning to where it's at its highest resolution in the end we can see
12:02
that the different parts of the image are much better segmented and that's how through one final convolution we get
12:09
this very clean mask yeah so units are really good at segmenting images there
12:14
was this International image segmenting competition where the people who invented the unit just went in there and
12:20
demolished everyone here's them getting the award for it what a bunch of nerds to be honest no I'm just kidding I mean
12:25
that in an endearing way but anyways okay when are we actually going to get to the image generation we're getting
12:31
there listen up the unet is so good at identifying things within an image that
12:37
people started using it for other stuff other than semantic segmentation specifically it could be used to de
12:42
noise an image if a noisy image is just the sum of the original image plus some

![image](https://github.com/user-attachments/assets/3d7091ce-401a-491a-b286-98bd1065273b)

![image](https://github.com/user-attachments/assets/e8f74740-f4a6-465e-acb8-62543935378b)


12:47
noise then if you identify the noise in the image then you can just minus it
12:53
away to get the original in fact that's exactly what we're going to try to do so allow me to demonstrate with another
12:59
image of a fish this time with a resolution of 64x 64 this time there is
13:05
no black and white ground truth mask to go with it instead we generate a bunch of noise to be our ground truth because
13:12
that's what we're trying to train the network to identify it's important that during training we train on many copies
13:18
of the image with different amounts of noise added in so that it's able to denoise really noisy images as well as
13:25
not so noisy ones and here's where an interesting challenge arises how do we provide the network with the knowledge
13:32
of how noisy each image sample is because that's obviously going to affect the outcome so if you imagine all the
13:39
possible noise levels placed in a sequence the information here of how
13:44
noisy any sample is is basically a number of that sample's position in that
13:50
sequence so this is called positional encoding so let's say for this particular image its noise level
13:56
corresponds to the 10th position in the sequence now we've got a 64x 64 image with three
14:02
channels meaning there's 12,288 numbers in total do we just slap
14:07
a 10 on the end making it 12,289 numbers is that going to work no so here's how
14:13
positional encoding works and I get it you might be thinking okay this seems like not such a significant detail why
14:19
do we need to go through it this might be like the fifth time I've said this but it's going to come up later again
14:25
it's going to be important positional encoding is a type of embed in which is when you take discrete variables like
14:32
Words hint later on or in this case positions in a sequence and turn it into
14:38
a vector of continuous numbers to feed to the network as a more digestible form
14:44
of information that it can then use the way that our 10 gets converted into a
14:49
vector of continuous numbers is using these s and cos equations here so that the vector of numbers always stays
14:55
within a fixed range but each position is encoded by a unique combination of numbers in the vector since the


