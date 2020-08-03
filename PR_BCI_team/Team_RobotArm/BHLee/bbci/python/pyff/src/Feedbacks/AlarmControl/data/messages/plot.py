#!/usr/bin/env python


import sys

import pylab, numpy



def main(filename):

    f = open(filename, "r")
    data = f.read()
    f.close()
    data = data.split("\n")
       
    t = 0
    data2 = {"red" : [],
            "yellow" : [],
            "green" : [],
            "grey": []}

    for i in data:
        if i == "":
            continue
        seconds, klass = i.split()
        seconds, klass = int(seconds), int(klass)
        t += seconds
        class2color = [None, "red", "yellow", "green", "grey"]
        data2[class2color[klass]].append([t, klass])

    pylab.plot([x for x, y in data2["red"]], [y for x,y in data2["red"]], "ro")
    pylab.plot([x for x, y in data2["yellow"]], [y for x,y in data2["yellow"]], "yo")
    pylab.plot([x for x, y in data2["green"]], [y for x,y in data2["green"]], "go")
    pylab.plot([x for x, y in data2["grey"]], [y for x,y in data2["grey"]], "ko")

    pylab.yticks( range(1,5), ("Rot", "Gelb", "Gruen", "Grau"))
    pylab.ylim(0, 5)
    pylab.xlabel("t [s]")
    pylab.savefig(filename.split(".")[0]+".png")

    for v in data2.itervalues():
        print len(v)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Usage: %s filename" % sys.argv[0]
        sys.exit(1)
    main(sys.argv[1])

