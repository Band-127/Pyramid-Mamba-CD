# Mamba Pyramid：

Our code is modified on the basis of ChangeMamba.

## 1 Set up
To start with, please follow the steps below:
~~~ bash
git clone []
conda create -n mambapyramid
cd MambaPyramid
conda activate mambapyramid
~~~
That is the way to create the env.

Then we install the requirements.
~~~ bash
pip install -r requirements.txt
cd kernels/selective_scan && pip install .
~~~

Please organize the datasets as follows:
~~~txt
--workspace(MambaPyramid)
    --data
        --LEVIR-CD
            -- train
                -- A
                -- B
                -- label
            -- test
                (same)
            -- val
                (same)
        --DSIFN-CD
            -- train
                -- mask
                -- A
                -- B
            (same)
        --SYSU-CD
            -- train
                -- time1
                -- time2
                -- lable
            -- test
                (same)
        --WHU-CD
            -- A
                -- whucd-00001.png
                ......
            -- B
                -- whucd-00001.png
                ......
            -- label
                -- whucd-00001.png
                ......
            -- list
                -- train.txt
                -- test.txt
                -- val.txt
~~~
Note that the train/val/test sets in WHU-CD are splited by *gen_train_val_test.py* in *list* folder.
You can find them in it. And please run it before your training and testing in case that raising error.


## 2 Train your model
For the 4 datasets above, we provide bash scripts for them in *scripts* folder.
~~~bash
    cd changedetection
    bash script/run/train_<dataset_name>.sh
~~~
Before you use them, please make sure that the data has been uploaded and the config path is correct.

Also, the evaluate the training process, you can see the matrics like train loss and corresponding metrics on Tensorboard.

## 3 Test your model
It is easy for you to get the test result of your model. We  provide two forms, the binary change map and the RGB difference map. For the former, you only need to change the   *if_visible* hyperparameter to 'gray' so you can get it. The white region stands for changed region, while the black stands for unchanged region. If you set the param as 'diff', you'll get the RGB maps. The red region stands for mistakes, while the green stands for misses.
~~~bash
    bash script/run/test_<dataset_name>.sh
~~~

The pre-trained model will be uploaded soon.

If the repository is useful for you, please tick a star and refer it in your paper.
