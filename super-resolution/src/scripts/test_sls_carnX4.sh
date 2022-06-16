python main.py --pretrained_dir ../experiment/$2/model/model_best.pt --gpu $1 --test_only --model carn --scale 4 --sls true --M 32
python main.py --pretrained_dir ../experiment/$2/model/model_best.pt --gpu $1 --data_test Set14 --test_only --model carn --scale 4 --sls true --M 32
python main.py --pretrained_dir ../experiment/$2/model/model_best.pt --gpu $1 --data_test B100 --test_only --model carn --scale 4 --sls true --M 32
python main.py --pretrained_dir ../experiment/$2/model/model_best.pt --gpu $1  --data_test Urban100 --test_only --model carn --scale 4 --sls true --M 32

