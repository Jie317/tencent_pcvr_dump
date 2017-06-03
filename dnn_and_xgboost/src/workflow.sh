

python3 main_DNN.pyp main_DNN.py -mlp -e 50 -v 2 -f 20 -s | tee log_ag_run

git add .
git commit -m 'finished run'
git pull
git push -f

sudo shutdown -a 0 

