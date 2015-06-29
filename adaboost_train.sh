
pencv_createsamples -info /home/liang/Desktop/vr/project3/info.txt -bg /home/liang/Desktop/vr/project3/bg.txt -vec pos.vec -w 20 -h 20 -num 50000

opencv_haartraining -data xml -vec pos.vec -bg bg.txt -nsplits 2 -mem 5120 -nosym -w 20 -h 20 -mode all -npos 5000 -nneg 11000

