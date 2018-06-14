import os
import subprocess

cmd = 'java -cp "$CLASSPATH" weka.classifiers.functions.SMOreg -C 1.0 -N 0 -classifications "weka.classifiers.evaluation.output.prediction.CSV -decimals 5" -I "weka.classifiers.functions.supportVector.RegSMOImproved -T 0.001 -V -P 1.0E-12 -L 0.001 -W 1" -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -t "./D2/lat_Train_case01.arff" -T "./D2/lat_Test_case01.arff" -c "1" -p 0'
# > ".\D2_case01_lat.csv"

#result = subprocess.check_output(cmd, stdout=subprocess.PIPE).stdout
result = subprocess.check_output(cmd, shell=True)

result_p = result[65:-2].replace('\n', ',').split(',')

for i in range(3, len(result_p), 4):
    print(result_p[i])

print(len(result_p))
