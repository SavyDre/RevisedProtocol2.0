import pathlib
import os
import shutil
import math
import pandas as pd

def writeOutputData(saveDir, datapath, classifiedDF, DetectionData, TestCenterData, quantDF, paperDict):
    """Write all output data to directory"""

    ## Drop Classification columns from the crDF and drDF dataframes
    DetectionData = DetectionData.drop("dr_Classification", axis=1)
    TestCenterData = TestCenterData.drop("tc_Classification", axis=1)

    ## Remove nan and fill with -
    classifiedDF.fillna(pd.NA, inplace=True)
    DetectionData.fillna(pd.NA, inplace=True)
    TestCenterData.fillna(pd.NA, inplace=True)

    ## Write data files
    classifiedDF.to_csv(pathlib.PurePath(datapath, "ClassifiedData.csv"), index=False)
    DetectionData.to_csv(pathlib.PurePath(datapath, "DetectionData.csv"), index=False)
    TestCenterData.to_csv(pathlib.PurePath(datapath, "TestCenterData.csv"), index=False)
    quantDF.to_csv(pathlib.PurePath(datapath, "PairedQuantificationData.csv"), index=False)

    # Writing paper dict variables to report
    try:
        print('Writing the paper dictionary to papervarstemplate')
        writeFile(saveDir, 'PaperVarsTemplate.tex', paperDict)
    except Exception as e:
        print(f'Could not write the paperVarsTemplate file due to exception: {e}')
    pass

def writeFile(outputFile, fileName, varsDictionary):
            # Create variable file
            varsScript = getVars(varsDictionary)
            # script = '\n'.join([headers, varsScript])
            # contents = '\n'.join([script, template])
            path = os.path.join(outputFile, fileName)
            with open(path, 'w') as f:
                # f.write(contents)
                f.write(varsScript)

def copyTemplate(dst, reportTemplate):
    cwd = os.getcwd()
    src = pathlib.PurePath(cwd, reportTemplate)
    shutil.copyfile(src, dst)

def getVars(vars):
    varList = ['WindSpeedAvgMin', 'WindSpeedAvgMax', 'WindSpeedAvgMean', 'TAtmAvgMin', 'TAtmAvgMax', 'TAtmAvgMean', 'RHavgmin', 'RHavgmax', 'RHavgmean', 'WDAvgMin', 'WDAvgMax', 'WDAvgMean'] # these columns should be added to the FiguresBuilder script: mimic Chiemezie's code. Also, change the naming
    script = ""
    for key, value in vars.items():
        key = str(key)
        key = key.replace('_', '')
        key = key.replace('1', 'One')
        key = key.replace('2', 'Two')
        key = key.replace('3', 'Three')
        key = key.replace('4', 'Four')
        key = key.replace('5', 'Five')
        key = key.replace('6', 'Six')
        key = key.replace('7', 'Seven')
        key = key.replace('8', 'Eight')
        key = key.replace('9', 'Nine')
        key = key.replace('0', 'Zero')
        # Todo: add more characters to replace
        if type(value) == str:
            value = value.replace('%', ' percent')
            value = value.replace('>', ' greater than ')
            value = value.replace('<', ' less than ')

        if type(value) == float:
            # Round the value to 3 sig figs
            if not math.isnan(value) and value != 0.0:
                value = round(value, 3 - int(math.floor(math.log10(abs(value)))) - 1)
        if key in varList:
            value = float(value)
            value = round(value, 3 - int(math.floor(math.log10(abs(value)))) - 1)
        script += ''.join(['\n\\newcommand{\\var', str(key), '}{', str(value), '}'])
    return script


