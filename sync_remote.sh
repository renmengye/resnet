FOLDER=code
NAME=resnet
rsync -rP \
--exclude "/$NAME/data" \
--exclude "/$NAME/results" \
--exclude "/$NAME/logs" \
--exclude ".git*" \
--exclude "*.pyc" \
../$NAME $USER@cs.toronto.edu:/u/$USER/$FOLDER 