python3 ffm_preprocess.py

./ffm-train -k 8 -t 5 -l 0.00003 ../data/pre/ffm_train ../data/pre/trained_ffm_model
./ffm-predict ../data/pre/ffm_test ../data/pre/trained_ffm_model results_from_ffm


cd ..
./quick_push.sh
