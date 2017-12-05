# Quickstart

On Mac OS, install Python 3, e.g.:
```
brew install python3
```

Fetch tutorial notebook, e.g:
```
mkdir ~/repos
cd ~/repos
git clone git@github.com:edgedown/alta17active.git
cd alta17active
```

Set up environment, e.g.:
```
virtualenv -p /usr/local/bin/python3 ve
. ve/bin/activate
pip install -r requirements.txt
```

Start Jupyter, e.g.:
```
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter notebook
```

This should open a jupyter tab in your web browser. If not, go to http://localhost:8888/notebooks/notebook.

Open exercise_1.1.ipynb and enjoy!
