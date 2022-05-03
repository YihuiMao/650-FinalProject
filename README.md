# 650-FinalProject
THE GITHUB LINK is https://github.com/YihuiMao/650-FinalProject

git clone git@github.com:YihuiMao/650-FinalProject.git

Because this project is based on the existing research what I am working on.
What I am doing on this project is looking for methods that improve the performance 
of our algorithem. Please look at my report. I may not to share the code with you guys.
our code only usually run in the cluster, and would be very difficult to set up in local
labtop, But I provide a colab code that does run my code. If you did need to run our code, 
please email me, i will shared the permission to you temporarily.  All files except for
tutorial.ipynb are for setup the env in colab. 

For cvae, login your seas email,go to the link 
https://drive.google.com/drive/folders/1Nt_hq4TX4c0QuaVA9fw5ND31TRLt6UiC?usp=sharing
download all csv files, and drag all files under cvae directory (/path/to/cvae) 
run python train.py, then you will see images in the directory of figs. (Shaoming Zheng)

For ensemble, run  (Yihui Mao)
python .\maze_dataset_1.py --dataset --num_rrt 1 --optimized
python .\maze_dataset_1.py --dataset --num_rrt 5 --optimized
python .\maze_dataset_1.py --dataset --num_rrt 1
python .\maze_dataset_1.py --dataset --num_rrt 1

the num_rrt is the num of the paths would create for ensemble model, (detail in the future work in report)
optimized is for smoothing the path.

the maze_dataset_.py would create training and testing set (images and file coordinates of start point and end points) 
for cvae, and also the dataset to give the sample (state and action )to dynamic model and the inverse model



 
