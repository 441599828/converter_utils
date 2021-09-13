#-*- coding: UTF-8 -*-
import json
import os
import sys
# 数据类型
# {
#  "name": "score.networkQuality",
#  "index": true,
#  "view": "app/views/score/networkQuality.tmpl.html",
#  "files": ["app/modules/controllers/score/networkQualityCtrl.js"]
# },
name = []
index = []
views = []
files = []
# "name": "monitored.monitored",
with open('/home/idriver/Downloads/perception_calibration(1).000000.json') as f:
 for line in f:
  try:
   line.index("name")
   # line = line.strip('\n')
   pos = line.index(':')
   name.append(line[pos+3:len(line)-3])
  except ValueError:
   pass
  try:
   line.index("true")
   # line = line.strip('\n')
   pos = line.index(':')
   index.append((line[pos+2:len(line)-2]))
  except ValueError:
   pass
  try:
   line.index("view")
   try:
    line.index("name")
   except ValueError:
    pos = line.index(':')
    views.append(line[pos + 3:len(line) - 3])
  except ValueError:
   pass
  # ['app/modules/controllers/monitored/monitoredCtrl.js","app/modules/services/resources/resourcesService.j']
  try:
   line.index("files")
   # line = line.strip('\n')
   pos = line.index(":")
   try:
    i=0
    str1 = ""
    str2 = ""
    str3 = ""
    pos_comma = line.index(",")
    str = line.split(',')
    comma_count = line.count(',')
    while i<=comma_count:
     if i == 0:
      # str[0] = str[0].strip('\n')
      str1 = str[0][pos+4:len(str[0])-1]
     elif i == 1 :
      if comma_count == 1:
       # str[1] = str[1].strip('\n')
       str2 = str1+','+str[1][1:len(str[1])-3]
      else:
       str2 = str1 + ',' + str[1][1:len(str[1]) - 1]
     elif i == 2 :
      # str[2] = str[2].strip('\n')
      str3 = str2+','+str[2][1:len(str[2])-3]
     else:
      pass
     i = i + 1
    if (comma_count == 1):
     files.append(str2)
    elif (comma_count == 2):
     files.append(str3)
   except ValueError:
    # line = line.strip('\n')
    files.append(line[pos+4:len(line)-3])
    pass
  except ValueError:
   pass
 if os.path.exists("C:\\Users\\sxf\\Desktop\data.txt"):
  os.remove("C:\\Users\\sxf\\Desktop\data.txt")
 write_file = open("C:\\Users\\sxf\\Desktop\data.txt","a+")
 j=0
 while j< len(name):
  str_info = name[j]+"\t"+index[j]+"\t"+views[j]+"\t"+files[j]+"\n"
  write_file.write(str_info)
  j = j + 1