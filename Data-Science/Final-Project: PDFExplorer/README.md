# PDFExplorer 

Let me introduce my **final project** of the data science bootcamp: The **PDFExplorer**

## Use Case 
During my studies in university, I worked as a student assistant to a professor who was a little bit disorganized. He instructed me to organize a giant folder of text files thematically. Needless to say, this was a very cumbersome task. 

At the end of the bootcamp I remembered this laborious task and decided to write a little python script that automatically organizes text files by similarity. I called this programm **PDFExplorer**. 

## How it works 

Without going into too much technical details, PDFExplorer works as follows. The basic, primitive but powerful, idea is that two texts are (more or less) similar to each other to the extend that they share words or phrases (bigrams or trigrams). (Obviously, this is only a very superficial description of how the program works, since there are numerous words (e.g. stop words) that are not relevant for the analysis of similarity).

Using sklearn's TfidfVectorizer, each text file becomes a point in an n-dimensional vector space (where n is the number of words or phrases picked out of the whole corpus). The similarity between the text files is thus simply the proximity of the points which represent said files in this vector space. This spatial proximity can be easily determined by using the cosine similarity metric. Finally, the grouping of different articles according to similarity can be easily achived by clustering (I used the KMeans algorithm). 

## Example 

You can find a video demonstrating the functionality of the PDFExplorer here:
https://youtu.be/qY-LfwcBmZI

**Note**: Using the PDFExplorer you can also visualize the different articles in a PCA plot. Note especially the bottom right corner. The two clusters "Justified Belief" and "Induction" are very close to each other yet seperated. This reflects the fact that both topics belong to epistemology! 


## Problems 
There are still a few problems with this script that I have not been able to fix because of a lack of time. Firstly, PDFExplorer is pretty slow. Secondly, it does not work so well with articles of a more mathematical nature due to the pdf encoding of mathematical symbols. Thirdly, I am not happy with the way I designed main.py, I wanted to create a true command line interface, but, again, time ran out and I had other projects to work on. 


## Dependencies 

You can find the dependencies in the requirements.txt file. 
