# Import modules
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, rc
# from IPython.display import HTML

# Import PySwarms
import pyswarms as ps
from prompt_toolkit import HTML
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher, Designer

if __name__ == '__main__':

    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=2, options=options)
    cost, pos = optimizer.optimize(fx.rosenbrock, iters=100)
    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
    rc('animation', html='html5')
    m = Mesher(func=fx.rosenbrock)

    # Make animation
    animation = plot_contour(pos_history=optimizer.pos_history,
                             mesher=m,
                             mark=(0, 0))

    # Enables us to view it in a Jupyter notebook
    HTML(animation.to_html5_video())

    # Obtain a position-fitness matrix using the Mesher.compute_history_3d()
    # method. It requires a cost history obtainable from the optimizer class
    pos_history_3d = m.compute_history_3d(optimizer.pos_history)
    # Make a designer and set the x,y,z limits to (-1,1), (-1,1) and (-0.1,1) respectively

    d = Designer(limits=[(-1, 1), (-1, 1), (-0.1, 1)], label=['x-axis', 'y-axis', 'z-axis'])

    # Make animation
    animation3d = plot_surface(pos_history=pos_history_3d,  # Use the cost_history we computed
                               mesher=m, designer=d,  # Customizations
                               mark=(0, 0, 0))  # Mark minima

    # Enables us to view it in a Jupyter notebook
    HTML(animation3d.to_html5_video())
    print('best score: ', cost, 'et position optimale : ', pos)
