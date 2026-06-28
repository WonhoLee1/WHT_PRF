import sys
from odbAccess import openOdb

if len(sys.argv) < 2:
    sys.exit("Usage: abaqus python extract_odb.py <job.odb>")

odb_name = sys.argv[1]
output_csv = odb_name.replace('.odb', '.csv')

odb = openOdb(odb_name)
step = odb.steps.values()[0]

with open(output_csv, 'w') as f:
    f.write("Time,LE11,LE22,LE33,LE12,S11,S22,S33,S12\n")
    for frame in step.frames:
        time = frame.frameValue
        
        if 'LE' not in frame.fieldOutputs or 'S' not in frame.fieldOutputs:
            continue
            
        le_field = frame.fieldOutputs['LE']
        s_field = frame.fieldOutputs['S']
        
        if len(le_field.values) == 0:
            continue
            
        le_val = le_field.values[0].data
        s_val = s_field.values[0].data
        
        # For 3D elements, data length is 6: 11, 22, 33, 12, 13, 23
        if len(le_val) >= 4:
            le11, le22, le33, le12 = le_val[0], le_val[1], le_val[2], le_val[3]
            s11, s22, s33, s12 = s_val[0], s_val[1], s_val[2], s_val[3]
            f.write("%g,%g,%g,%g,%g,%g,%g,%g,%g\n" % (time, le11, le22, le33, le12, s11, s22, s33, s12))
        else:
            # For 2D elements
            le11, le22, le33, le12 = le_val[0], le_val[1], le_val[2], le_val[3] if len(le_val)>3 else 0.0
            s11, s22, s33, s12 = s_val[0], s_val[1], s_val[2], s_val[3] if len(s_val)>3 else 0.0
            f.write("%g,%g,%g,%g,%g,%g,%g,%g,%g\n" % (time, le11, le22, le33, le12, s11, s22, s33, s12))

odb.close()
