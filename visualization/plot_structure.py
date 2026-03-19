import matplotlib.pyplot as plt


def plot_truss(nodes,elements,displacements=None,scale=1):

    for e in elements:

        n1,n2,_,_ = e

        x1,y1 = nodes[n1]
        x2,y2 = nodes[n2]

        plt.plot([x1,x2],[y1,y2],'k--')

    if displacements is not None:

        new_nodes = {}

        for i,(x,y) in nodes.items():

            ux = displacements[2*i]*scale
            uy = displacements[2*i+1]*scale

            new_nodes[i] = (x+ux,y+uy)

        for e in elements:

            n1,n2,_,_ = e

            x1,y1 = new_nodes[n1]
            x2,y2 = new_nodes[n2]

            plt.plot([x1,x2],[y1,y2],'r')

    plt.axis('equal')
    plt.show()
