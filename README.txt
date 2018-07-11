README
=======



All of the code is written in python and KERAS. You need to install KERAS.

Then unpack the ZIP file in a directory.

CD into that directory. It will be called ForHuawei.

The file you are interested in is plearn2Hua.py.

You can just run it using this command.

python plearn2Hua.py

It will train our network and then perform the validation using our data in the HUAWEI directory.


If you open the python code in an editor go to the bottom of the file.
You will see


if True:
   perform_training(...
   perform_validation(...


If False:
   ...
   ...
   peform_genetic_evolution(...



This is a simple way to run the code. If you change the code to this    


if False:
   perform_training(...
   perform_validation(...


If True:
   ...
   ...
   peform_genetic_evolution(...


You will just run the genetic algorithm (GA). You can play with the parameters in the function.
If you set generations=8, population=30 and nepochs=100 it will take about 5 hours to run on a TitanV GPU.
These are the parameters we used to generate our final network.

The current parameters in the code will run the GA in about 60 seconds.

