f1 = open("../train_pre_1")
f2 = open("../test_pre_1")
out1 = open("../train_pre_1b","w")
out2 = open("../test_pre_1b","w")
t = open("../train_gbdt_out")
v = open("../test_gbdt_out")
add = []
for i in range(30,49):
	add.append("C" + str(i))

line = f1.readline()
print(line[:-1] + "," + ",".join(add), file=out1)
line = f2.readline()
print(line[:-1] + "," + ",".join(add), file=out2)
for i in range(40428967):
	line = f1.readline()[:-1]
	a = t.readline()[:-1]
	ll = a.split(" ")[1:]
	for j in range(19):
		line += "," + add[j] + "_" + ll[j]
	print(line, file=out1)
for i in range(4577464):
	line = f2.readline()[:-1]
	a = v.readline()[:-1]
	ll = a.split(" ")[1:]
	for j in range(19):
		line += "," + add[j] + "_" + ll[j]
	print(line, file=out2)

f1.close()
f2.close()
out1.close()
out2.close()
t.close()
v.close()
