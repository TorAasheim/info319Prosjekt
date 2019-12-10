When first trying to run the program, pycharm will mention all of the packs that needs to be installed, this can be done with pycharm:

Orca should be installed via Anaconda.
If orca is installed in the main folder of Anaconda3, copy orca.cmd to envs/learning
orca: https://github.com/plotly/orca


If the computer has 8gb or less of RAM, JVM needs to be allocated more memory than default.
Navigate to sparkfolder/conf edit file spark-default.conf.template by making a copy of the fil and removing .template from the file extension.
Uncomment the line spark.driver.memory and set it to "spark.driver.memory              5g".

Running the program:

First, run the file called stemming.py
Then run  Bayes.py.

For the best result run it in an IDE with a console. 
We recommend Pycharm as that is what we developed it in and if any dependencies are missing, Pycharm will notify you.
When done correctly the output of the program will be a .png file that visualizes the sentiment for each month, and a .csv file in the prediction folder that shows the prediction for the whole dataset. 

potential errors:


-Make sure that in the environment variables Java_Home is pointing to jdk8
- If "ValueError: For some reason, plotly.py was unable to communicate with the local orca server process, even though the server process seems to be running":
This can be a timeout error, which we have experienced with one of our slow machines. This only affects the visualization, but the machine-learning has been completed, so we have attached a copy of the visualization in case.
- The folder "predictions" will have to be deleted if the program is to be run after it's cration