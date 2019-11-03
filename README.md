## SECTION 1 : PROJECT TITLE
## Esports Object Detection with MobilenetSSD
<img src="Miscellaneous/title picture.png"
     style="float: left; margin-right: 0px;" />
---
## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT
In this project, we will train an image object detection model to detect computer graphic objects in popular esports computer games.

We are interested in this topic as object detection has been used in many real life competitive sports already such as soccer. Using object detection as input, sports teams and betting companies can analyse the match better. For example, one can track the distribution of players on pitch, movements of balls and changes in formations. However, such techniques have not been used as much in Esports.


Esports are getting more and more popular in recent years. Competitive games like League of Legends, PUBG, CS Go and Dota attract millions of players and audience all over the world. We believe that this type of vision systems will be useful in esports. It will help game analyst and online streaming channels(youtube, twitch) provide real time analytics. It can also help team to understand pro-player’s playing style and other teams’ strategy. We believe such system will actually be easier to train and have better detection performance because in-game graphics are generated by computer drawing. Thus, unlike real world pictures, they should be more systematic in color and style, for example all houses in the game look similar as most likely they are copied and pasted. In-game real time analytics will also be more reliable as camera angles of the game are usually limited and fixed.

---
## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name  | Student ID (MTech Applicable)  | Work Items (Who Did What) |
| :------------ |:---------------:| :-----|
| Chan Kan Hei | A0198512Y | Data acquisition, data preprocessing, opencv conversion|
| Guo Xiang  | A0198533U | Data preprocessing, code integration, model training|
| Li Jingmeng | A0198484J | Datat preprocessing, model training, model evaluation|

---
## SECTION 4 : USER GUIDE

### [ 1 ] To train the model

> Go to SourceCode/ and run the first cell of TrainTest.ipynb.

> You can change the arguments according to your needs.

### [ 2 ] To test the model

> Run the second cell of TrainTest.ipynb.

> You can run the cell directly using the weights we provided in Miscellaneous/checkpoints/.

---
## SECTION 5 : PROJECT REPORT / PAPER

`<Github File Link>` : <https://github.com/telescopeuser/Workshop-Project-Submission-Template/blob/master/ProjectReport/Project%20Report%20HDB-BTO.pdf>

---
## SECTION 6 : MISCELLANEOUS

### checkpoints/
* Directory to save the checkpoints during training.
* ssd300_epoch-200.h5 is the weights we trained previously. You can use it as weights to test the model.

### data/
* Data repository.
* Inside test_images/  there are some sample images for you to test the model.

### weights/
* Stores the weights of Mobilenet.
