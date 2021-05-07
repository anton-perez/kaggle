import time
start = time.time()

counter = 0
for _ in range(1000000):
  counter += 1

end = time.time()
print(end - start)

#0.29506587982177734
#relative speed = 2