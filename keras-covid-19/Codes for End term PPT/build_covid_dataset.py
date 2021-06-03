import shutil
import os
import pandas as pd
import argparse



arg = argparse.ArgumentParser()
arg.add_argument("-c", "--covid", required=True,
	help="dataset path")
arg.add_argument("-o", "--output", required=True,
	help="normal images path")
args = vars(arg.parse_args())


csvPath = os.path.sep.join([args["covid"], "metadata.csv"])
df = pd.read_csv(csvPath)


for (i, row) in df.iterrows():
	
	if row["finding"] != "COVID-19" or row["view"] != "PA":
		continue

	
	imagePath = os.path.sep.join([args["covid"], "images",
		row["filename"]])

	
	if not os.path.exists(imagePath):
		continue

	
	filename = row["filename"].split(os.path.sep)[-1]
	outputPath = os.path.sep.join([args["output"], filename])

	# copy the image
	shutil.copy2(imagePath, outputPath)