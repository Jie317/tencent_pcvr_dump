echo "Start"
date
# python3 ffm_preprocess.py

./ffm-train -k 8 -t 10 -l 0.00003 -s 4 ../ffm_train ../trained_ffm_model 


echo "Predict"
./ffm-predict ../ffm_test ../trained_ffm_model ../results_from_ffm

date
python3 write_to_submission.py


# ./run_ffm_pcvr.sh | tee ../run_log_`date +"%m%d_%H%m"`