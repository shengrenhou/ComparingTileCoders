import json
import sys 
import os 
# print(sys.argv)
path=os.getcwd()
print(path)

# get current work dictory 
with open ("c:\\Users\\hshengren\\Documents\\GitHub\\ComparingTileCoders\\experiments\\compare\\MountainCar\\SARSA.json","r") as f:
    data=json.load(f)
    print(data)

