touch /home/iad35/scalable_gps/tmp_file.txt
python gp_experiment_runner.py -m ./model_specs/RBF_model_spec.json -d pumadyn32nm -o run_outputs/RBF_test_run.csv -s 0.1 -r 2 --device cuda --no_cv  >& run_outputs/test_run_stdout.txt 
python gp_experiment_runner.py -m ./model_specs/ARD_model_spec.json -d pumadyn32nm -o run_outputs/ARD_test_run.csv -s 0.1 -r 2 --device cuda --no_cv  >& run_outputs/test_run_stdout.txt 

