FOLDER=code
NAME=resnet
rsync -rP \
--exclude "/$NAME/data" \
--exclude "/$NAME/results" \
--exclude "/$NAME/logs" \
--exclude "/$NAME/.git" \
--exclude "/$NAME/.gitignore" \
../$NAME $USER@cs.toronto.edu:/u/$USER/$FOLDER 