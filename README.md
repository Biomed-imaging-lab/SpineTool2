## Install
1. Download code
2. Unzip CGAL.zip next to code, e.g. `PATH_TO_CODE\CGAL\...`
3. Install [Anaconda](https://www.anaconda.com/)
4. Open Anaconda
5. Execute
```cmd
cd PATH_TO_CODE
conda create --name spinetool2 --file conda_requirements.txt -y
conda activate spinetool2
pip install -r pip_requirements.txt
pip install -r plugins/ai_segmentation/requirements.txt 
pip install easy3d-2.6.1-cp310-cp310-win_amd64.whl
```
6. if use cuda, execute
```cmd
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade --force-reinstall
```
in not use cuda, execute
```cmd
pip install torch
```

## Run
1. Open Anaconda
2. Execute
```cmd
cd PATH_TO_CODE
conda activate spinetool2
python run.py
```

# Run ai segmentation and vsot segmentation
1. install gnu octave: download https://octave.org/download zip, extract
2. clone vsot main branch https://github.com/yu-lab-vt/VSOT 
3. to run predictions faster, install cuda driver + cuda toolkit
4. download stage 1 to 4 segmentation models  and place to  plugins\ai_segmentation\models\stage_1\,  plugins\ai_segmentation\models\stage_2\, plugins\ai_segmentation\models\stage_3\, plugins\ai_segmentation\models\stage_4\
5. run
```cmd
cd PATH_TO_CODE
conda activate spinetool2
python run.py
```
6. set neural segmentation settings: path to vsot root (e.g. `H:\Programs\VSOT-main`), path to vsot matlab dir (e.g. `H:\Programs\VSOT-main\src\+comSeg`) path to octave executable exe (e.g. `H:\Programs\octave-11.3.0-w64\octave-11.3.0-w64\mingw64\bin\octave-cli.exe`)


# Common errors

## cuda usage

RuntimeError: CUDA requested but is not available
steps:
1. check cuda availability manually
```
PATH_TO_ANACONDA\Anaconda\envs\spinetool2\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```
2. result should be like:
```
2.x.x+cu...
12.x
True
NVIDIA GeForce RTX 2060
```
3. if output contains None/ False - check torch version, it should have postfix `cu`, like `2.4.1+cu124`, reinstall torch (switch off vpn)


RuntimeError: Numpy is not available
steps:
1. run 
```cmd
pip install --force-reinstall "numpy==1.26.4"
```