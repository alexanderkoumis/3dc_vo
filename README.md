# 3DC-VO

This is the source code used for the paper "Estimating Metric Scale VO from Videos". It includes code to train rotation and translation models, and a script to evaluate the models against the validation dataset.

## Paper result reproduction

There are several steps involved in reproducing the paper results.

 1. Download dataset
 2. Run script to generate trajectories used for evaluation
 3. View trajectories using some third party tool

### Getting the dataset
First, download the resized (160x90) KITTI odometry dataset in the releases section of this repo, and unzip it here: `$THIS_REPO/data/images/`

### Running trajectory generation script

```bash
./src/create_results_file.py --reproduce
```

This will evaluate the pretrained models (at `$THIS_REPO/data/models/model_pretrained_(yaw|y).h5`) on validation sequences 11-21 (the ones to submit to KITTI's leaderboard). It will create files `11.txt`, `12.txt`, and so on, under `$THIS_REPO/output/`.

### View trajectories

You can either plot the trajectories in the .txt files yourself or use another tool that has this functionality built in, such as Michael Grupp's [evo](https://github.com/MichaelGrupp/evo). For example, using evo, you can view the generated trajectory with the following command:

```bash
evo_traj kitti $THIS_REPO/output/13.txt --plot_mode=xz -p
```

This will show a plot like this, which is one of the trajectories used in the paper:

![Sequence 13](https://i.imgur.com/EyG2Z2Y.png)


## Training your own model

You can either use your own images or use the dataset used in the paper. To use the dataset in the paper, download the dataset and place it in the correct directory as described above in "Getting the dataset". To use your own dataset, make sure it has the same folder structure as the one included in this release.

### Training step
You must train two separate models (rotational and translational). To train the rotational model, run the training script as follows:

```bash
mkdir -p $THIS_REPO/data/models/yaw
DATA_DIR=$THIS_REPO/data/images
MODEL_FILE=$THIS_REPO/data/models/yaw/model_yaw.h5
./src/train.py $DATA_DIR $MODEL_FILE yaw
```

Then, train the translational model as follows:
```bash
mkdir -p $THIS_REPO/data/models/y
DATA_DIR=$THIS_REPO/data/images
MODEL_FILE=$THIS_REPO/data/models/yaw/model_y.h5
./src/train.py $DATA_DIR $MODEL_FILE y
```

### Evaluation step

To evaluate your generated models, you must compile the official KITTI odometry evaluation tool, included in this repo under `$THIS_REPO/src/third_party/kitty_eval`:

```bash
cd $THIS_REPO/src/third_party/kitty_eval
g++ -std=c++11 -O3 -DNDEBUG -o evaluate_odometry_quiet ./evaluate_odometry_quiet.cpp ./matrix.cpp
```

This is a slightly modified version of the official tool, which is much less verbose and only evaluates validation sequences instead of train and validation sequences.

Then, run the included evaluation script, which will run the compiled evalution tool on your trained models, and display them in a sorted order based on how accurate they were:

```bash
EVAL_BIN=$THIS_REPO/src/third_party/kitty_eval/evaluate_odometry_quiet
DATA_DIR=$THIS_REPO/data/images
MODEL_DIR=$THIS_REPO/data/models/yaw
YAW_OR_Y=yaw
MODEL_PRETRAINED=$THIS_REPO/data/models/model_pretrained_y.h5
./src/eval_models.py $EVAL_BIN $DATA_DIR $MODEL_DIR $YAW_OR_Y $MODEL_PRETRAINED
```
This will look for the highest performing rotation model, using the pretrained y network to estimate forward offsets. To find the highest performing trainslation model, run the script with the following arguments:

```bash
EVAL_BIN=$THIS_REPO/src/third_party/kitty_eval/evaluate_odometry_quiet
DATA_DIR=$THIS_REPO/data/images
MODEL_DIR=$THIS_REPO/data/models/y
YAW_OR_Y=y
MODEL_PRETRAINED=$THIS_REPO/data/models/model_pretrained_yaw.h5
./src/eval_models.py $EVAL_BIN $DATA_DIR $MODEL_DIR $YAW_OR_Y $MODEL_PRETRAINED
```

## Questions?
Feel free to email me at me@alexander.computer.
