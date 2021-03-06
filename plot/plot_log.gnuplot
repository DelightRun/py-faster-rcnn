# These snippets serve only as basic examples.
# Customization is a must.
# You can copy, paste, edit them in whatever way you want.
# Be warned that the fields in the training log may change in the future.
# You had better check the data files before designing your own plots.

# Please generate the neccessary data files with 
# /path/to/caffe/tools/extra/parse_log.sh before plotting.
# Example usage: 
#     ./parse_log.sh mnist.log
# Now you have mnist.log.train and mnist.log.test.
#     gnuplot mnist.gnuplot

# The fields present in the data files that are usually proper to plot along
# the y axis are test accuracy, test loss, training loss, and learning rate.
# Those should plot along the x axis are training iterations and seconds.
# Possible combinations:
# 1. Test accuracy (test score 0) vs. training iterations / time;
# 2. Test loss (test score 1) time;
# 3. Training loss vs. training iterations / time;
# 4. Learning rate vs. training iterations / time;
# A rarer one: Training time vs. iterations.

# What is the difference between plotting against iterations and time?
# If the overhead in one iteration is too high, one algorithm might appear
# to be faster in terms of progress per iteration and slower when measured
# against time. And the reverse case is not entirely impossible. Thus, some
# papers chose to only publish the more favorable type. It is your freedom
# to decide what to plot.

reset
set terminal postscript eps color
set datafile sep ','

set output "RPNClsLoss.eps"
set style data lines
set key right

# lcs loss vs. training iterations
set title "RPN cls loss vs. training iterations"
set xlabel "Training iterations" font ",20"
set ylabel "Cls loss" font ",20"
plot "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage1.train" using 1:4 title "Stage 1" lc rgb 'red', "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage2.train" using 1:4 title "Stage 2" lc rgb 'blue'

set output "RPNBBoxLoss.eps"
set style data lines
set key right

# cls loss vs. training iterations
set title "RPN bbox loss vs. training iterations"
set xlabel "Training iterations" font ",20"
set ylabel "BBox loss" font ",20"
plot "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage1.train" using 1:5 title "Stage 1" lc rgb 'red', "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage2.train" using 1:5 title "Stage 2" lc rgb 'blue'

set output "RCNNClsLoss.eps"
set style data lines
set key right

# lcs loss vs. training iterations
set title "R-CNN cls loss vs. training iterations"
set xlabel "Training iterations" font ",20"
set ylabel "Cls loss" font ",20"
plot "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rcnn_stage1.train" using 1:4 title "Stage 1" lc rgb 'red', "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rcnn_stage2.train" using 1:4 title "Stage 2" lc rgb 'blue'

set output "RCNNBBoxLoss.eps"
set style data lines
set key right

# lcs loss vs. training iterations
set title "R-CNN bbox loss vs. training iterations"
set xlabel "Training iterations" font ",20"
set ylabel "BBox loss" font ",20"
plot "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage1.train" using 1:5 title "Stage 1" lc rgb 'red', "faster_rcnn_alt_opt_ZF_.txt.2016-05-15_13-56-25_rpn_stage2.train" using 1:5 title "Stage 2" lc rgb 'blue'
