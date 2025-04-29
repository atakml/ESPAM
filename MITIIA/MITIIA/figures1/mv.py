import os
import shutil
for layer in range(5):
    median_files = os.listdir(".")
    for f in median_files:
        if f[-11:] == "_median.png":
            dir_name = f[14:-11]
            print(f"mv *\"_{dir_name}\"* \"{dir_name}\"")
            os.system(f"mkdir \"{dir_name}\"")
            os.system(f"mv *\"_{dir_name}\"* \"{dir_name}\"")
        '''os.system(f"mv {layer}_median_final/*\"_{dir_name}\"* \"{dir_name}\"")
        os.system(f"mv {layer}_random_final/*_\"{dir_name}\"* \"{dir_name}\"")
        os.system(f"mv {layer}_random/*\"_{dir_name}\"* \"{dir_name}\"")'''
