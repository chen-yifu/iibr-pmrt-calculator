#!/bin/bash

REPO_PATH="/Users/yifu/Documents/github-repo/iibr-pmrt-calculator/"

cd $REPO_PATH
git pull
git commit --allow-empty -m "Empty commit to keep app awake"
git push
