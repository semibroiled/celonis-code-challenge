# celonis-code-challenge

A Multi Classification Problem to Recognize Gestures from Accelerometer Time Series Data

Celonis Code Challenge Task 1 

Abstract

Due to lack of skill and experience, I'll try to keep the complexity as low as possible and just to convey the working paradigms and decision designs behind the project, as would have been normally done in the interview. The challenge is meant to be kept to 3-4 hour so I hope the following is sufficient in its breadth.

For the project, I used the numpy, pyplot and os libraries only.

For better project management I used self made functions as modules

prototype.ipynb = is the Jupyter notebook I used to test and try the different approached to working with the regression. Since I've never implemented a regression from scratch before, I decided to get my hands dirty by trying out a linear regression. As my experience later would reinforce, this was a lot of learning and relearning as I realized the skills in pythong and maths needed is something I possessed to start with, but I was inexperienced in how to combine them. Much of that has been deleted, but I did use the scripts I used to access the txt files and make the X Matrix features and y target vector, all of which I later put into functions for ease of use. Later, I would go on to implement a logisitic regression model, a multi classification variant, through help from university scripts, friend and some online resources. I tried to follow the math and in some cases read the documentation in sklearn and numpy libraries to try to keep the task as grounded as possible. I try to hint to my inspirations as well as I can in the comments.

Since the task is due to be submitted today on friday within working hours, I'm avoiding burning more midnight oil by attempting the optional 
exercises, things which I am already familiar with working through online resources.
Rather I focus on the implementation with numpy aspect, something which I haven't done before 
and also presents a novel challenge

i tried to use as type hinting as much as possible and used scales and Black
to keep my code clean and readable. I understand however that it might still be 
convovulated and quite far from best practices.

I tried to use type hinting as much as possible on advice from my fnf.
This was not always possible with some functions as typing couldn't be imported
in VS Code. I tried to ensure as much docstring documentation as possible


There were some solutions i considered such as loading the features in a different way (multi dimensional arrays),
interpolating data instead of padding with 0, using other coordinate axes etc. But due to time crunch I decided to keep it as ideas
for later implementation for my own use, as they would raise complexity of concept.

For reasons unknown to me the normalization is raising errors. This isn't working with any of the fixes I know of.
Maybe I'll leave it out of the final solution and just present it as an area I need further work on.

The One Hot Column in numpy was handled quite nicely. The Numpy documentation and sklearn documentation
helped a lot for this. There was the idea of using the column numbers to
implement unit testing with pytest on github. I don't think I'll come around to doing that due to time constraints.

Also for the logistical regression, I chose to forego forward gradient ascent too much, instead
relying on the sigmoid fucntion. As this was the approach taken by the Springer Book
on Numerical Analysis (2012) 

I was made aware that my memory management with numpy could be much more efficient and smart.

As I intended to work with low iterations to begin with i did not necessarily choose
hyper optiized methods, but also I was limited by my capacity. As such, the speeds are quite slow 
when compared to its load.


Test = There was the idea to implement unit testing with pytest and CLI  but these have been decided against to limit time

