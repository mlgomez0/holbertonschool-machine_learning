#0x01. Plotting

## Concepts

- What is a plot?
- What is a scatter plot? line graph? bar graph? histogram?
- What is matplotlib?
- How to plot data with matplotlib
- How to label a plot
- How to scale an axis
- How to plot multiple sets of data at the same time

#Installation
Files were interpreted/compiled on Ubuntu 16.04 LTS using python3 (version 3.5)
Files were executed with numpy (version 1.15)
Files will be executed with numpy (version 1.15) and matplotlib (version 3.0)

## Usage

Educational purposes

## Tasks
0. Line Graph: 0-line.py
```
Complete the following source code to plot y as a line graph:

y should be plotted as a solid red line
The x-axis should range from 0 to 10
```
1. Scatter: 1-scatter.py
```
Complete the following source code to plot x  y as a scatter plot:

The x-axis should be labeled Height (in)
The y-axis should be labeled Weight (lbs)
The title should be Men's Height vs Weight
The data should be plotted as magenta points
```
2. Change of scale: 2-change_scale.py
```
Complete the following source code to plot x  y as a line graph:

The x-axis should be labeled Time (years)
The y-axis should be labeled Fraction Remaining
The title should be Exponential Decay of C-14
The y-axis should be logarithmically scaled
The x-axis should range from 0 to 28650
```
3. Two is better than one: 3-two.py
```
Complete the following source code to plot x  y1 and x  y2 as line graphs:

The x-axis should be labeled Time (years)
The y-axis should be labeled Fraction Remaining
The title should be Exponential Decay of Radioactive Elements
The x-axis should range from 0 to 20,000
The y-axis should range from 0 to 1
x  y1 should be plotted with a dashed red line
x  y2 should be plotted with a solid green line
A legend labeling x  y1 as C-14 and x  y2 as Ra-226 should be placed in the upper right hand corner of the plot
```
4. Frequency: 4-frequency.py
```
Complete the following source code to plot a histogram of student scores for a project:

The x-axis should be labeled Grades
The y-axis should be labeled Number of Students
The x-axis should have bins every 10 units
The title should be Project A
The bars should be outlined in black
```
5. All in One: 5-all_in_one.py
```
Complete the following source code to plot all 5 previous graphs in one figure:

All axis labels and plot titles should have a font size of x-small (to fit nicely in one figure)
The plots should make a 3 x 2 grid
The last plot should take up two column widths (see below)
The title of the figure should be All in One
```
6. Stacking Bars: 6-bars.py:
```
Complete the following source code to plot a stacked bar graph:

fruit is a matrix representing the number of fruit various people possess
The columns of fruit represent the number of fruit Farrah, Fred, and Felicia have, respectively
The rows of fruit represent the number of apples, bananas, oranges, and peaches, respectively
The bars should represent the number of fruit each person possesses:
The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
Each fruit should be represented by a specific color:
apples = red
bananas = yellow
oranges = orange (#ff8000)
peaches = peach (#ffe5b4)
A legend should be used to indicate which fruit is represented by each color
The bars should be stacked in the same order as the rows of fruit, from bottom to top
The bars should have a width of 0.5
The y-axis should be labeled Quantity of Fruit
The y-axis should range from 0 to 80 with ticks every 10 units
The title should be Number of Fruit per Person
```
