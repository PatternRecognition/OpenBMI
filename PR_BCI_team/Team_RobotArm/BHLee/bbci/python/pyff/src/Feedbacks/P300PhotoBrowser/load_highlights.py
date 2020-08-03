import random

def load2(filename, subtrial_count, highlight_count):
	subtrial_count = int(subtrial_count)
	highlight_count = int(highlight_count)
	fp = open(filename, "r")
	data = fp.readlines()
	fp.close()

	lines = []
	totalvalues = 0
	for d in data:
		dt = d[:-1] # remove the \n
		vals = dt.split(',')
		print len(vals)
		print (subtrial_count * highlight_count)
		if len(vals) != (subtrial_count * highlight_count):
			print "Error: line %d has incorrect number of values!" % (len(lines)+1)
			return None
		totalvalues += len(vals)
		lines.append(vals)

	print "(Loading highlights for %d trials)" % (len(lines))

	entries = []

	for i in range(len(lines)):
		entries.append([])
		for j in range(subtrial_count):
			entries[i].append([])
			for k in range(highlight_count):
				entries[i][j].append(int(lines[i][(j*highlight_count)+k]))

	return entries

def save(filename, trial_count, subtrial_count, highlight_count):
	subtrial_count = int(subtrial_count)
	highlight_count = int(highlight_count)
	s = ""
	for i in range(trial_count):
		for j in range(subtrial_count):
			for k in range(highlight_count):
				s+=str(random.randint(0, 30))+","
		s = s[:-1]
		s += '\n'
	fp = open(filename, "w")
	fp.write(s)
	fp.close()
