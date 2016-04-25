import matplotlib.pyplot as plt
import sys

def foo(data):
    """
    data: a list of lists or tuples(whatever) that contains the coordination of the curve
    data[i][0] is x coord, data[i][1] is y coord
    """
    sum = 0
    for i in range(len(data)-1):
        area = (data[i][1] + data[i+1][1])*(data[i+1][0]-data[i][0]) / 2
        sum += area

    return sum

def read_data(filename):
    file = open(filename, 'r')
    ret = []
    for line in file:
        word = [w for w in line[:-1].split(' ') if w != '']
        if word[0] == 't':
            t = float(word[1])
        elif word[0] == 'TPR':
            TPR = float(word[1])
            FPR = float(word[3])
            ret += [(t,TPR,FPR)]
    file.close()
    ret.sort(key=lambda x:x[2])
    return ret

def draw(argv):
    """
    if len(argv)==2:
        if argv[1] == 'p':
            filename = 'priori_roc'
        elif argv[1] == 'r':
            filename = 'random_roc'
        elif argv[1] == 'n':
            filename = 'naive_roc'
        else:
            filename = 'random_roc'
    else:
        print("parameter format: p|r|n")
        sys.exit()
"""
    
    random = read_data('random_roc')
    priori = read_data('priori_roc')
    naive = read_data('naive_roc')

    sum_r = foo([(x[2],x[1]) for x in random])
    sum_n = foo([(x[2],x[1]) for x in naive])
    sum_p = foo([(x[2],x[1]) for x in priori])

    print(sum_n, sum_r, sum_p)

    plot_r, = plt.plot([a[2] for a in random], [a[1] for a in random], 'b-')
    plot_p, = plt.plot([a[2] for a in priori], [a[1] for a in priori], 'g-')
    plot_n, = plt.plot([a[2] for a in naive], [a[1] for a in naive], 'k-o')

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend([plot_p, plot_r, plot_n], ['priori', 'random', 'naive'], 'best', numpoints=1)
    plt.show()
    
"""
    plot1, = plt.plot(x,p1,'r')
    plot2, = plt.plot(x,p2,'b')
    plot3, = plt.plot(x,p3,'y')
    plt.xlabel('subproblem no. of NA')
    plt.ylabel('F1')
    plt.legend([plot1,plot2,plot3],['a=1','a=2','a=3'],'best',numpoints=1)
    #plt.legend([],'a=1','a=2','a=3')
    plt.show()
"""
if __name__ == '__main__':
    draw(sys.argv)