							 
nohup python -u inout_svm.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes=<num of classes>  --io_perc=--io_perc=<percentage of displacement e.g. 0.5  is 50% outward> --print_pos=1 > log/log1 2>&1
nohup python -u inout_svm.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes=<num of classes>  --io_perc=--io_perc=<percentage of displacement e.g. -0.5 is 50% inward> --print_pos=2 > log/log2 2>&1

 
