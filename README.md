# paper_DA

This is the repository for the code of the paper: 
"On the Effectiveness of Data Augmentation Strategies 
for Different Types of Domain Shifts and Learning Approaches"

Given the images for the data sets/domains

launch the following to generate the domains splits  
python gen_split.py --data_path=<path to daomain> --num_classes=<number of classes> --random_state=<random_seed>

launch the following to generate the embeddings  
python gen_embs.py --embs_path=<path to embs> --num_classes=<num of classes> --strat=N<random_seed>

launch the following to experiment with in-out (fully connected) strategy  
python -u inout_fcl.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes==<num of classes>  --io_perc=<percentage of displacement e.g. -0.5  is 50% inward> --print_pos=2 > log/log2 2>&1

launch the following to experiment with in-out (SVM) strategy  
python -u inout_svm.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes=<num of classes>  --io_perc=--io_perc=<percentage of displacement e.g. -0.5 is 50% inward> --print_pos=2 > log/log2 2>&1

launch the following to experiment with rotation (fully connected) strategy  
python -u rotat_fcl.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes=<nnum od classes> --ang_rad=<angle in radians> --print_pos=2  > log/log_2  2>&1

launch the following to experiment with rotation (svm) strategy  
python -u rotat_svm.py --data_path_dom1=<path to dom1 embeddins> --data_path_dom2=<path to dom2 embeddins> --num_classes=<num of classes>  --ang_rad=<angle in radians> --print_pos=1  > log/log_1  2>&1
