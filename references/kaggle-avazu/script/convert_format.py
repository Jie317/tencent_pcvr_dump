import os
import sys

def convert_format(path):
	os.rename(path, '%s_tmp' % path)
	with open('%s_tmp' % path, 'r') as inp, open(path, 'w') as outp:
		for l in inp:
			tmp = l.split()
			label = tmp[0]
			f1 = tmp[1:30]
			f2 = tmp[30:]
			str_f1 = ['0:%d:%s'%((i+1), v) for i,v in enumerate(f1)]
			str_f2 = ['1:%d:%s'%((i+1+len(f1)), v) for i,v in enumerate(f2)]

			str_tmp = '%s\t%s\t%s\n'%(label, '\t'.join(str_f1), '\t'.join(str_f2))
			outp.write(str_tmp)


convert_format(sys.argv[1])
convert_format(sys.argv[2])
