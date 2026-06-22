# MidJourney 
by Abhinav Gupta
https://www.youtube.com/watch?v=rSOd2-SBsXQ


#### What is MidJourney
is a Text Based AI Art Generator
Version 1 was released in Feb 2022. The current version is 5.2 

### Getting started 
* Download Discord.com ( https://docs.midjourney.com/docs/midjourney-discord) 
* Create a new Server 
* Add APP 
* Search for Midjourney bot 
* Add the bot to server
  

## Subscribe 
/Subscribe

## Easy Prompts
* Start with a Basic Idea
* Build on top of it by adding new variables and detailing your prompt
* be more specific (avoid ambiguity)


#### /Imagine
is the basic prompt script. 

```bash
/imagine prompt Tom cruise in manyavar ethnic clothes
```
![image](https://github.com/user-attachments/assets/18e65915-9e07-4129-8ec6-afe48d2aab50)

#### U1,U2,U3, U4 
are the images which can be selected for further updating or optimizing called upscaling

#### V1, V2, V3, V4
are the variables that can be chnaged 

*Highlight* 
**what**
/*this*/ 
***this****

## Medium Prompts
this is my using **advance variables** which gives more control over the composition of the images or how the image should feel or look like. check with Nick St. Pierre from insta for variables

### Variables

#### Artistic Style

* Studio Gibley
* Retro
* punk
* Editorial 

#### Composition

* Wide shot
* fully body
* Ariel
* bird eye shot 

#### Median

* Sketch
* Screen grab
* Sclpture
  
#### Film Type or Camera
specify camera type say example fuji film camera 

* Lens Type (24 mm etc. )
* Aperture
* focus
* bokeh effect 

#### Subject Description
basic prompt like clothes, facial expression etc.

#### Environment 
what is the subject doing. location of the subject 

* Underwater
* War zone

#### Lighting 

* Night light
* day Light
* Studio light
* dramatic
* natural 

#### Atmosphere

* thunder stormy
* stormy
*  Normal sunlight
*  foggy

#### Mood
* Luxurious
* romantic
* dreamy

#### Examples 

```bash
# Style
editorial stype, tom cruise in manyar ethnic clothes

1990s punk style, indian actor ajith with cigar with gold rolex watch in hand

one piece anime style, Rajinikath as minkey d luffy in one piece anime

# Composition
editorial style ultrawide drone shot of indian farmer using futuristic drone to do farming

editorial style extreme low angle camera shot of indian farmer using futuristic drone to do farming

# Median
concept art line sketch, india metro

watercolor, indian metro

# Camera
will smith doing ganesh pooja, kodak Vision 200T Negative Film 5274/7274

will smith doing ganesh pooja, canon eos 5d mark 4

```

## Tips 

* In MidJourney website go to explore and copy the prompts and use it
* 

## Hard 


#### --no commnad 

--no trees greenery nature plants

#### Prompt weights 
emphasis certain parts of the image. default 1 
```bash
aerial shot of futuristic::2 india metro

aerial shot of futuristic::3 india metro::2

```

#### Multi prompt 
use :: 

```bash
#this is simple thought
cricket bat
#now lets use multiple prompts. two thoughts but same emphasis 
cricket:: bat
#now lets use multiple prompts. two thoughts but cricket has more/double emphasis 
cricket::2 bat
```
#### aspect ration --ar
```bash
#basic
green trees, photorealistic --ar 9:16
```

#### seed
gives consistency with prompt continuity

where to find the seed or the id of the image
Click on upscale--> add reaction --> envelop --> select it 

or you can assign random number yourself

![image](https://github.com/user-attachments/assets/1e473f10-c961-4c69-b2ca-228aeaafe26f)

![image](https://github.com/user-attachments/assets/e3c4a708-a546-4e15-924d-406c79ed0264)

```bash
#basic
green trees, photorealistic --ar 9:16 --seed <number>
```

#### Stylize 
varies between 0-1000

```bash
#basic
a futurstic mushroom --seed 101 -- stylize 100
```

#### chaos 
how much variation should be there in 4 generated images 
values from 0-100; default is 0 i.e. more consistent 

```bash
#basic
a futurstic mushroom --chaos 0

#Extreme
a futurstic mushroom --chaos 100
```

#### Repeat 
number of times the 4 images should be generated. if given 5, then 20 variation will be generated. 

```bash
#basic
a futurstic mushroom -- stylize 100 --ar 9:16 --repeat 5
```

# Cool stuff 

#### scaling the images. use "U" and still you want scaling use the site  
[waifu2x](https://waifu2x.udp.jp/)

#### face swap
InsightFace bot add to discord

```bash
#download a image to be swapped and save in computer 
#Add image to prompt 
/saveid idname <SRK> image <upload>
/listid
/setid idname <SRK>
#go to any generated image thats needs to be swapped --> LT Click --> Apps --> INSwapper  
```

#### Blend 
combines 2 or more images and create a new one 

```bash
#save 2 images thats needs to blend on computer
#basic
/blend image1 <upload> image2 <upload> 
```

#### refrence image 
give a refrence image for asthetic/style and then generate 
use some images from artstation 

```bash
#click on prompt and then upload image. rt click and then copy link 
/imagine prompt <link> <skyscrapper in city)
```

#### Anime 
use --niji command

```bash
#basic
/imagine prompt <boy riding a bycycle in india> --niji
```

#### logo 

```bash
#basic
/imagine prompt <a flat logo for a coffee brand, white background, startup, minimal>
# download the logo you like as jpeg
# upload to vectorizer.ai and chose options and then download
```

# Usefull links

Official Midjourney [website](https://www.midjourney.com/home/)
Resource for artistic styles:

[Convert to SVG](https://vectorizer.ai/)
artstation [artstation](https://www.artstation.com/?sort_by=community&dimension=all)
Artworks by style - WikiArt.org: https://www.wikiart.org/en/paintings-...
[V4 Midjourney Reference Sheets - Google Sheets](https://docs.google.com/spreadsheets/d/1MsX0NYYqhv4ZhZ7-50cXH1gvYE2FKLixLBvAkI40ha0/edit?gid=274691973#gid=274691973)
[Theory of Prompting](https://ww7.newcomputerscience.com/theory?usid=24&utid=10511066088)
[PromptHero - Search prompts for Stable Diffusion, ChatGPT & Midjourney](https://prompthero.com/)
Excel file [Prompter – Prompter Guide](https://prompterguide.com/prompter/)
[Library of Midjourney artist styles - Google Sheets] ([https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1hTYW54a19iampZZndJdEtwblJTekdoZ013UXxBQ3Jtc0tsd3R6UFV0dUhFZDNaNHAwUXJmVG4xa282WklnZC1YQkkwNlVKZXFmc1RuLVZXbU9CZEhKb1FzbTJlUUFZMFFvbHNGX2pKUFh4VjZwX25pM2h6Qmw0bzBOMzFrbUdSd29Cd1ZLOGtsUHA0MzFWa0M1VQ&q=https%3A%2F%2Fdocs.google.com%2Fspreadsheets%2Fd%2F1cm6239gw1XvvDMRtazV6txa9pnejpKkM5z24wRhhFz0%2Fedit%3Fpli%3D1%23gid%3D400749539&v=rSOd2-SBsXQ](https://docs.google.com/spreadsheets/d/1cm6239gw1XvvDMRtazV6txa9pnejpKkM5z24wRhhFz0/edit?pli=1&gid=400749539#gid=400749539)

[Midjourney Prompt Generator v5 - artificin.com](https://artificin.com/prompt-builder)

Twitter People to follow:

Nick St. Pierre (@nickfloats):   / nickfloats  
Julie W. Design (@juliewdesign_):   / juliewdesign_  
Linus (●ᴗ●) (@LinusEkenstam):   / linusekenstam  
Kris Kashtanova (@icreatelife):   / icreatelife  
Óscar Bartolomé (@Artedeingenio):   / artedeingenio  
phil desforges (@storybyphil):   / storybyphil  
Pierrick Chevallier | IA (@CharaspowerAI):   / charaspowerai  
TechHalla (@techhalla):   / techhalla  
















