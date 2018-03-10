# ASL Translation System Application

## Requirements
- python >= 3.5
- the rest is the `api/requirements.txt`

## Good to know (and what I already know)
- first prediction is very heavy for a PC, so it can take a long time
- some glitches occurs in the hand position detection display on Ubuntu 16.04
- `CSS` of the application is not adjusted for every display - many windows were made just for a debugging and `CSS` is not a strong side of mine

## Installation and running

First installation: 
```angular2html
git clone https://github.com/kacper1095/translation-system-application.git
cd translation-system-application
pip install -r api/requirements.txt
npm install
```

Running:
```angular2html
./run_system.sh
```

*If you get error with `electron` word inside stack trace, try running following command first:
```bash
sudo npm install -g electron --unsafe-perm=true --allow-root
```