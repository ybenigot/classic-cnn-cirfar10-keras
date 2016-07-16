timestamp=`date +%d%m%y-%T`
filename=record/test$timestamp.log
touch $filename
python convnet.py 100 $timestamp >> $filename 2>&1 & 
tail -f $filename

