python main.py -d configure/data.yaml -m configure/tfnet.yaml --mode train -n 5 # train and evaluation on independent test set
python main.py -d configure/data.yaml -m configure/tfnet.yaml --mode eval -n 5 # evaluate on test set
python main.py -d configure/data.yaml -m configure/tfnet.yaml --mode predict -n 5 # predict on independent data set

python main.py -d configure/data.yaml -m configure/tfnet.yaml --mode 5cv # 5 cross-validation
python main.py -d configure/data.yaml -m configure/tfnet.yaml --mode lomo # leave one data out cross-validation

#baseline model
python baseline.py configure/data.yaml