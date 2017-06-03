#! /bin/bash

# Script to samplify the commit and push steps. All the arguments are concatenated as the commmit message.

git add -A
git status
if [ $# -eq 0 ]; then
	echo "Warning: Using default message: update"
	echo ""
	git commit -m "Update"
else
	git commit -m "$*"
fi
git push 

# Other tips
# 1 ssh login: https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/ then run: ssh-keyscan -t rsa 
# github.com >> ~/.ssh/known_hosts
# 2 init a repo: 
# git init && git remote add origin git@github.com:Jie317/***.git
# git config --global user.name "Your name here"
# git config --global user.email "your_email@example.com"
