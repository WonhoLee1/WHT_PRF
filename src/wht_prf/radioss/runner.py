import os
import subprocess
import logging

def run_openradioss(run_dir, run_name, exec_dir):
    """
    Runs OpenRadioss starter and engine.
    exec_dir: e.g. "D:/OpenRadioss_win64/OpenRadioss/exec"
    """
    starter_exe = os.path.join(exec_dir, "starter_win64.exe")
    engine_exe = os.path.join(exec_dir, "engine_win64.exe")
    
    # Setup OpenRadioss Environment
    or_root = os.path.dirname(exec_dir)
    env = os.environ.copy()
    
    paths_to_add = [
        exec_dir,
        os.path.join(or_root, "extlib", "intelOneAPI_runtime", "win64"),
        os.path.join(or_root, "extlib", "hm_reader", "win64"),
        os.path.join(or_root, "extlib", "h3d", "lib", "win64")
    ]
    env["PATH"] = ";".join(paths_to_add) + ";" + env.get("PATH", "")
    env["KMP_STACKSIZE"] = "400m"
    env["RAD_CFG_PATH"] = os.path.join(or_root, "hm_cfg_files")
    
    # 1. Run Starter
    starter_cmd = [starter_exe, f"-i", f"{run_name}_0000.rad"]
    logging.info(f"Running Starter: {' '.join(starter_cmd)}")
    
    try:
        starter_proc = subprocess.run(starter_cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=True)
        logging.info("Starter completed successfully.")
        print(f"Starter STDOUT:\n{starter_proc.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Starter failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        print(f"Starter failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False
        
    # 2. Run Engine
    engine_cmd = [engine_exe, f"-i", f"{run_name}_0001.rad"]
    logging.info(f"Running Engine: {' '.join(engine_cmd)}")
    
    try:
        engine_proc = subprocess.run(engine_cmd, cwd=run_dir, env=env, capture_output=True, text=True, check=True)
        logging.info("Engine completed successfully.")
        print(f"Engine STDOUT:\n{engine_proc.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Engine failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        print(f"Engine failed!\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False
        
    return True
