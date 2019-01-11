# face-capture

### Setup

* Install the [Miniconda Python 3.x](https://conda.io/miniconda.html) version corresponding to the development 
operating system.
* Clone this repository:
```
git clone https://github.com/thefonseca/face-capture.git
cd face-capture
```

* Create the conda environment (this should be done only once):
```
conda env create -f environment.yml
```
* Activate the conda environment:
```
source activate face-capture
```

### Detect faces in camera frames
```
python detect_faces_camera.py
```

### Detect faces in Youtube video
```
python detect_faces_youtube.py -v cUw18IAkmvE
```

### Detect and save faces in videos of a Youtube playlist
```
python detect_faces_youtube.py -p PLitz1J-q25kMvoRT9AIPfWoFIv4wzoUPi --playlist-start 1 -s 20
```
