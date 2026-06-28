from odbAccess import openOdb
import sys

try:
    odb = openOdb('test_prony_val.odb')
    step = odb.steps.values()[0]

    s11_data = []
    le11_data = []

    for r_name, region in step.historyRegions.items():
        if 'Int Point 1' in r_name:
            if 'S11' in region.historyOutputs:
                s11_data = region.historyOutputs['S11'].data
            if 'LE11' in region.historyOutputs:
                le11_data = region.historyOutputs['LE11'].data
            break

    with open('test_prony_val_extracted.txt', 'w') as f:
        for s, e in zip(s11_data, le11_data):
            f.write(str(s[0]) + ',' + str(s[1]) + ',' + str(e[1]) + '\n')
    odb.close()
except Exception as e:
    with open('test_prony_val_error.txt', 'w') as f:
        f.write(str(e))
