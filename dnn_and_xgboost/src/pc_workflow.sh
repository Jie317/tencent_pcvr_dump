python3 -u main_DNN.py -m mlp_fe -mt 0 -v 2 -s | tee log_fe_with_ui_ua_and_ajusted_emb_dim
python3 -u main_DNN.py -m mlp_fe -mt 1 -v 2 -s | tee -a log_fe_with_ui_ua_and_ajusted_emb_dim
python3 -u main_DNN.py -m mlp_fe -mt 2 -v 2 -s | tee -a log_fe_with_ui_ua_and_ajusted_emb_dim
python3 -u main_DNN.py -m mlp_fe -mt 3 -v 2 -s | tee -a log_fe_with_ui_ua_and_ajusted_emb_dim
python3 -u main_DNN.py -m mlp_fe -mt 4 -v 2 -s | tee -a log_fe_with_ui_ua_and_ajusted_emb_dim
