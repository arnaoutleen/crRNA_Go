# Overview
crRNA Go is a a machine-learning-based web application that can help predict the location of energetically-favorable binding sites for Cas13b on an ssRNA viral genome, and associated energies of such locations. Such ML application should be able to be trained by the user. The web-app will use these sequences to design a crRNA template that can be ordered and used in the lab. 

This repository has the code for the flask web-app that you can download, set up and execute on your localhost.

For the back-end python scripts, please go to my other repo: https://github.com/arnaoutleen/crRNA_Go_backend

For a video guide about using the app: https://youtu.be/u8__MzhsM4U


# To better understand how this program works, read these slides
https://docs.google.com/presentation/d/10fOaQTVwgwUEoDLApQ1MJ4ZB8ncaO7QaIXrIc_JAOlc/edit?usp=sharing


# Contents

|

|-```README.md```: obvious what it does

|-```Report.pdf```: report

|-```code```:

    |-```load_test_data.txt```: link to where test_data.csv is online, to download it
  
    |-``` my_model_bidirectional.h5```: sample ML model
    
    |-``` flaskapp.py```: flask app
    
    |-``` static```: folder for static files (they'll be saved here)
    
    |-``` pycache```
    
    |-``` templates```: folder for HTML front-end templates
    
        |-``` designer.html```: designer page
        
        |-``` train.html```: training page
        
        |-``` train_results.html```: training results page
        
        |-``` binding site finder.html```: binding site finder page
        
        |-``` binding site results.html```: binding site finder results page


# Co-Dependencies
 to run, this project requires: Python 3.7.6, Flask 1.1.1, Werkzeug 1.0.0, TensorFlow 2.2.0, Keras 2.3.1, Pandas 1.0.1, PIL/pillow 7.1.2, numpy 1.18.1, Biopython 1.77 (this includes Bio.Seq and Bio.SeqFeature and Bio.Graphics), sklearn 0.0, time

 It was built and run on Ubuntu 18.04


# How to run (also in report)
Download all the files into a root directory. The root directory should include all the .py files along with a folder for "Test Files". Download the test_data.csv files as instructed in "load_test_data.txt" and add it to root directory. Navigate to root directory and open terminal window there. Then execute these commands to run website on localhost (127.0.0.1:5000):

``` export FLASK_APP=1
export FLASK_DEBUG=1
flask run
```

If you’re running the app locally, you need to redefine UPLOAD_FOLDER in flaskapp.py to the root directory where you are hosting the flask app, and the static folder. This is in line 28 of flaskapp.py. In the sample code uploaded it is in: '/home/cornelia/Desktop/flaskapp'. Also, a WSGI debugger must be installed on your system such that you can debug the app while running it. If you do not wish to debug, you are free to not install the WSGI debugger, and the only other thing you don’t need to do is that you do not need to run this line of code: FLASK_DEBUG=1.

# License
I mean, this was an app developed for a college class so ... yeah use it in whatever way you'd like just please credit me and link to this repo so that I can see what people do with this.
