python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'comparison opt' --va --opt adagrad -v 2 | tee log_comparison_va_opt
python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'continue train opt' --va --opt adagrad -v 2 --ct | tee -a log_comparison_va_opt
python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'comparison without va' --opt adagrad -v 2  | tee -a log_comparison_va_opt

python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'comparison opt' --va --opt adam -v 2 | tee -a log_comparison_va_opt
python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'continue train opt' --va --opt adam -v 2 --ct | tee -a log_comparison_va_opt
python3 -u main_DNN.py -m mlp_fe -f 16 --ne --mess 'comparison without va' --opt adam -v 2  | tee -a log_comparison_va_opt

echo "Workflow finished" >> log_comparison_va_opt
