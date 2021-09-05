# Find RL_Note path and append sys path
import os, sys

def set_path(dir_name):
  cwd = os.getcwd()
  tmp1 = cwd.lower()
  tmp2 = dir_name.lower()
  pos = tmp1.find(tmp2)
  root_path = cwd[0:pos] + dir_name
  sys.path.append(root_path)
  return root_path
