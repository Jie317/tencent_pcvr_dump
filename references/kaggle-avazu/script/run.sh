cd ../gbdt
make
cd ../script

python3 addc.py
python3 fcount.py
python3 rare.py
python3 id_day.py
python3 prep.py
python3 id_stat.py
python3 gbdt_dense.py
python3 index1.py
python3 index2.py
../gbdt/gbdt -d 5 -t 19 ../test_dense ../train_dense ../test_gbdt_out ../train_gbdt_out

# fm model 1
python3 append_gbdt.py

python3 convert_format.py ../fm_test_2 ../fm_train_2 # for test only

../fm/fm -k 8 -t 5 -l 0.00003 ../fm_test_2 ../fm_train_2

# fm model 2
python3 append_gbdt_1.py

python3 convert_format.py ../fm_test_2_1 ../fm_train_2_1 # for test only
../fm/fm -k 8 -t 4 -l 0.00004 ../fm_test_2_1 ../fm_train_2_1

python3 convert_format.py ../fm_test_2_2 ../fm_train_2_2 # for test only
../fm/fm -k 8 -t 10 -l 0.00005 ../fm_test_2_2 ../fm_train_2_2

python3 split.py ../fm_test_2_split ../fm_test_2_1.out ../fm_test_2_2.out

# ftrl model prepare
python3 prep_1.py
python3 append.py
python3 genDict.py
python3 genM.py
python lsa.py

# ftrl model 1
python3 ftrl_1.py

# ftrl model 2
python3 ftrl_2.py

# ensemble
python3 ensemble.py