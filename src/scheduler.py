import os
with open('schedule.txt') as f:
    for line in f:
        os.system(line)