"""
WHTOOLs MATCALIB 2026
--------------------------------------------------------------------------------
Abaqus ODB Extraction Utility
Extracts Time, Stress (S11), and Strain (LE11/E11) from Abaqus ODB files into TXT.

Usage (Abaqus Command Line):
    abaqus python extract_odb.py <job_name.odb>
--------------------------------------------------------------------------------
"""
import sys
from odbAccess import openOdb

def extract_odb(odb_path):
    odb = openOdb(odb_path)
    out_file = odb_path.replace('.odb', '_abaqus_results.txt')
    
    with open(out_file, 'w') as f:
        f.write("Time\tS11\tE11\n")
        total_time = 0.0
        
        # Determine the first part instance
        instance_name = odb.rootAssembly.instances.keys()[0]
        
        for step_name in odb.steps.keys():
            step = odb.steps[step_name]
            for frame in step.frames:
                time = frame.frameValue # if multiple steps, Abaqus accumulates time in totalTime? No, frameValue is step time.
                # Actually, there is frame.description but step time is fine if we add total_time.
                current_total_time = total_time + frame.frameValue
                
                s11 = 0.0
                e11 = 0.0
                
                try:
                    s_field = frame.fieldOutputs['S']
                    s_val = s_field.values[0].data[0]
                    s11 = s_val
                except Exception as e:
                    pass
                
                try:
                    e_field = frame.fieldOutputs['LE']
                    e_val = e_field.values[0].data[0]
                    e11 = e_val
                except:
                    try:
                        e_field = frame.fieldOutputs['E']
                        e_val = e_field.values[0].data[0]
                        e11 = e_val
                    except:
                        pass
                
                f.write("{}\t{}\t{}\n".format(current_total_time, s11, e11))
                
            total_time += step.timePeriod
            
    odb.close()
    print("Extracted to " + out_file)

if __name__ == "__main__":
    odb_path = sys.argv[-1]
    extract_odb(odb_path)
