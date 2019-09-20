#!/usr/bin/env bash
sudo apt-get install libproj-dev proj-data proj-bin
sudo apt-get install libgeos-dev

virtualenv -p python3 deep_crater_env
source deep_crater_env/bin/activate
read -p "Install packages? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
pip install h5py
pip install Keras==2.0.0
pip install numpy
pip install opencv-python==3.2.0.6
pip install pandas==0.19.1
pip install Pillow
pip install scikit-image==0.12.3
pip install tables==3.4.2
pip install cython
pip install cartopy
pip install jupyter
echo installation complete
fi
echo configure jupyter:
jupyter notebook --generate-config
nano ~/.jupyter/jupyter_notebook_config.py
# add these lines to the file, save and exit:
#c = get_config()
#c.NotebookApp.ip = '*'
#c.NotebookApp.open_browser = False
#c.NotebookApp.port = 7000

#launch jupyter:
#jupyter-notebook --no-browser --port=7000 &

