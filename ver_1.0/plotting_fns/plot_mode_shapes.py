import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

from plotting_fns.drawing_tools_3d import *
from utilities.utils import *





def plotModeShape2D(_nodes, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'black', 'linestyle':'solid'}, kwargs_plot_markers={'color':'black', 'marker':'o'}):
    num_frames = _connectivity.shape[0]
    for k in range(num_frames):
        x1 = _nodes[_connectivity[k, 0] - 1, 0]
        y1 = _nodes[_connectivity[k, 0] - 1, 1]
        x2 = _nodes[_connectivity[k, 1] - 1, 0]
        y2 = _nodes[_connectivity[k, 1] - 1, 1]
        xx = [x1, x2]; yy = [y1, y2]
        ax.plot(xx, yy, zorder=10, **kwargs_plot_lines)
    if kwargs_plot_markers['marker']!=None:
        for i in range(_nodes.shape[0]):
            xs = _nodes[i, 0]; ys = _nodes[i, 1]
            ax.scatter(xs, ys, zorder=10, **kwargs_plot_markers)
            if annotFlag:
                ax.annotate(f'P{i + 1}', xy=(xs, ys), xytext=(3, 3),
                            textcoords='offset points',zorder=10)
            
def plot_modes_for_2d(modes_mean, modes_std, freq_mean, _nodes, _connectivity, connectivity_mode_shape_dofs, \
                      MODESCALEFCT, MODESTDFCT, MODE2D_DIRECTION, RESULTS_PATH):
    for ii in range(modes_mean.shape[0]):
        fig,ax = plt.subplots(figsize=(6,5),facecolor='white')
        plotModeShape2D(_nodes, _connectivity, ax, annotFlag=True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        nodes_new = np.copy(_nodes)
        nodes_new_infStd = np.copy(nodes_new)
        nodes_new_supStd = np.copy(nodes_new)
        tmp_STD_DEV = np.zeros_like(_nodes)
        for jj in range(modes_mean.shape[1]): # cycle on mode shape vector columns and associate mode shape to nodes
            tmp_id = np.where(connectivity_mode_shape_dofs == jj)
            nodes_new[tmp_id] += MODESCALEFCT * modes_mean[ii,jj]
            nodes_new_infStd[tmp_id] = nodes_new[tmp_id]
            nodes_new_supStd[tmp_id] = nodes_new[tmp_id]
            nodes_new_infStd[tmp_id] -= MODESTDFCT * modes_std[ii,jj]
            nodes_new_supStd[tmp_id] += MODESTDFCT * modes_std[ii,jj]
            tmp_STD_DEV[tmp_id] = modes_std[ii,jj]
        plotModeShape2D(nodes_new, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'#023e7d', 'linestyle':'solid'}, kwargs_plot_markers={'color':'#023e7d', 'marker':'o'})
        plotModeShape2D(nodes_new_infStd, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'#9bd0d1', 'linestyle':'dashed'}, kwargs_plot_markers={'color':'#023e7d', 'marker': None})
        plotModeShape2D(nodes_new_supStd, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'#9bd0d1', 'linestyle':'dashed'}, kwargs_plot_markers={'color':'#023e7d', 'marker': None})
        plt.title(f'Mode shape at {freq_mean[ii]:.3f} Hz', fontweight='bold')
        custom_lines = [
                    Line2D([0], [0], color='black',linestyle='dashed', lw=4),
                    Line2D([0], [0], color='#023e7d', lw=4),
                    Line2D([0], [0], color='#9bd0d1',linestyle='dashed', lw=4),
                    ]
        plt.legend(custom_lines,['Undeformed shape','Deformed shape',f'{MODESTDFCT:d} Std. dev.'],loc='best')
        plt.ylim(plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]+0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.png', format='png', dpi=300)
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.pdf')
        plt.close()
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}.txt', nodes_new, fmt='%.4f') 
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}_STD_DEV.txt', tmp_STD_DEV) 

def plot_modes_for_2d_for_paper(modes_mean, modes_std, freq_mean, _nodes, _connectivity, connectivity_mode_shape_dofs, \
                                MODESCALEFCT, MODESTDFCT, MODE2D_DIRECTION, RESULTS_PATH):
    for ii in range(modes_mean.shape[0]):
        fig,ax = plt.subplots(figsize=(6,5),facecolor='white')
        plotModeShape2D(_nodes, _connectivity, ax, annotFlag=True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        nodes_new = np.copy(_nodes)
        if MODE2D_DIRECTION == 'vertical' :
            nodes_new[1:,0] += MODESCALEFCT * modes_mean[ii,:]
            nodes_new_infStd = np.copy(nodes_new)
            nodes_new_supStd = np.copy(nodes_new)
            nodes_new_infStd[1:,0] -= MODESTDFCT * modes_std[ii,:]
            nodes_new_supStd[1:,0] += MODESTDFCT * modes_std[ii,:]
            plotModeShape2D(nodes_new, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'#023e7d', 'linestyle':'solid'}, kwargs_plot_markers={'color':'#023e7d', 'marker':'o'})
            ax.fill_betweenx(nodes_new[:,1],nodes_new_infStd[:,0],nodes_new_supStd[:,0],
                            color='#9BD0D1',
                            zorder=1,
                            label=f'{MODESTDFCT:d} Std. dev.')
        elif MODE2D_DIRECTION == 'horizontal':
            nodes_new[:,1] += MODESCALEFCT * modes_mean[ii,:]
            nodes_new_infStd = np.copy(nodes_new)
            nodes_new_supStd = np.copy(nodes_new)
            nodes_new_infStd[:,1] -= MODESTDFCT * modes_std[ii,:]
            nodes_new_supStd[:,1] += MODESTDFCT * modes_std[ii,:]
            plotModeShape2D(nodes_new, _connectivity, ax, annotFlag=False, kwargs_plot_lines={'color':'#023e7d', 'linestyle':'solid'}, kwargs_plot_markers={'color':'#023e7d', 'marker':'o'})
            ax.fill_between(nodes_new[:,0],nodes_new_infStd[:,1],nodes_new_supStd[:,1],
                            color='#9BD0D1',
                            zorder=1,
                            label=f'{MODESTDFCT:d} Std. dev.')
        else:
            raise ValueError("Error in direction of mode shapes. Accepted arguments: 'horizontal' or 'vertical' ")

        plt.title(f'Mode shape at {freq_mean[ii]:.3f} Hz', fontweight='bold')
        custom_lines = [
                    Line2D([0], [0], color='black',linestyle='dashed', lw=4),
                    Line2D([0], [0], color='#023e7d', lw=4),
                    Patch(facecolor='#9BD0D1', edgecolor=None)
                    ]
        plt.legend(custom_lines,['Undeformed shape','Deformed shape',f'{MODESTDFCT:d} Std. dev.'],loc='best')
        plt.ylim(plt.gca().get_ylim()[0],plt.gca().get_ylim()[1]+0.5)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.png', format='png', dpi=300)
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.pdf')
        plt.close()
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}.txt', nodes_new, fmt='%.4f') 
        tmp_STD_DEV = np.zeros_like(_nodes)
        for jj in range(modes_mean.shape[1]): # cycle on mode shape vector columns and associate mode shape to nodes
            tmp_STD_DEV[np.where(connectivity_mode_shape_dofs == jj)] = modes_std[ii,jj]
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}_STD_DEV.txt', tmp_STD_DEV) 
















def plotModeShape3D(nodes, connectivity, annotatepoints=False, figsize=(5,6), hold_on=False, kwargs_plot_lines={'color':'black', 'linestyle':'solid'}, kwargs_plot_markers={'color':'black', 'marker':'none'}):
    num_frames = connectivity.shape[0]
    if hold_on:
        _fig=plt.gcf()
        ax=plt.gca()
    else:
        _fig = plt.figure(figsize=figsize,facecolor='white')
        ax = plt.axes(projection="3d")
        setattr(ax, 'annotate3D', annotate3d)
    for k in range(num_frames):
        x1 = nodes[connectivity[k, 0] - 1, 0]
        y1 = nodes[connectivity[k, 0] - 1, 1]
        z1 = nodes[connectivity[k, 0] - 1, 2]
        x2 = nodes[connectivity[k, 1] - 1, 0]
        y2 = nodes[connectivity[k, 1] - 1, 1]
        z2 = nodes[connectivity[k, 1] - 1, 2]
        xx = [x1, x2]; yy = [y1, y2]; zz = [z1, z2]
        ax.plot3D(xx, yy, zz, **kwargs_plot_lines)
    for i in range(nodes.shape[0]):
        xs = nodes[i, 0]; ys = nodes[i, 1]; zs = nodes[i, 2]
        ax.scatter(xs, ys, zs, **kwargs_plot_markers)
        if annotatepoints:
            ax.annotate3D(ax,text=f'P{i + 1}', xyz=(xs, ys, zs), xytext=(3, 3), textcoords='offset points')
    return _fig,ax


def plot_modes_for_3d(modes_mean, modes_std, freq_mean, _nodes, _connectivity, connectivity_mode_shape_dofs, MODESCALEFCT, RESULTS_PATH):
    for ii in range(modes_mean.shape[0]):
        fig,ax = plotModeShape3D(_nodes, _connectivity, kwargs_plot_lines={'color':'gray', 'linestyle':'dashed'})
        ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')
        nodes_new = np.copy(_nodes)
        tmp_STD_DEV = np.zeros_like(_nodes)
        for jj in range(modes_mean.shape[1]): # cycle on mode shape vector columns and associate mode shape to nodes
            tmp_id = np.where(connectivity_mode_shape_dofs == jj)
            nodes_new[tmp_id] += MODESCALEFCT * modes_mean[ii,jj]
            tmp_STD_DEV[tmp_id] = modes_std[ii,jj]
        fig,ax = plotModeShape3D(nodes_new, _connectivity, hold_on=True, kwargs_plot_lines={'color':'#023e7d', 'linestyle':'solid'}, kwargs_plot_markers={'color':'#023e7d', 'marker':'o'})
        plt.title(f'Mode shape at {freq_mean[ii]:.3f} Hz', fontweight='bold')
        custom_lines = [
                    Line2D([0], [0], color='gray',linestyle='dashed', lw=4),
                    Line2D([0], [0], color='#023e7d', lw=4),
                    ]
        plt.legend(custom_lines,['Undeformed shape','Mode shape'],loc='best')
        plt.tight_layout()
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.png', format='png', dpi=300)
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.pdf')
        pickle.dump(fig, open(RESULTS_PATH + f'/Mode_shape_{ii+1}.pkl','wb'))
        plt.close()
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}.txt', nodes_new, fmt='%.4f') 
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}_STD_DEV.txt', tmp_STD_DEV) 

        # Show the figure, edit it, etc.!
        # figx = pickle.load(open(RESULTS_PATH + f'/Mode_shape_{ii+1}.pkl', 'rb'))
        # axx = plt.gca()












def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', center=None, **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # print(f'pearson={pearson}')
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    # print(mean_x,mean_y)
    
    if center:
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(center[0], center[1])
    else:
        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return pearson

def plot_ellipses(_nodes, _connectivity, connectivity_mode_shape_dofs, modes_mean, Frequency_dataset, \
                  MODESTDFCT_ELLIPSES, MODESCALEFCT, RESULTS_PATH):
    # Frequency_dataset is a list with length equal to the number of founded modes, each containing arrays
    # [ ['Frequency', 'Order', 'Label', 'Damp', 'Emme', 'ModeNum', 'SimNumber'], ['SimNumber','dof','dof','...'] ]
    Pearson_coeff=[] 
    # Pearson_coeff list of 3d arrays of pearson coefficients related to index 0 and 1 row and column of graphs respectively for each floor,
    # index 2 pearson coefficient related to A1, A2 and A3
    sensors_loc = import_data('DATA/Sensors_locations_tower.txt')
    for ii in range(modes_mean.shape[0]): # ii: cycle on frequency dataset cluters
        nodes_new = np.copy(_nodes)
        # tmp_MEAN_MODE = np.zeros_like(_nodes)
        for jj in range(modes_mean.shape[1]): # cycle on mode shape vector columns and associate mode shape to nodes
            tmp_id = np.where(connectivity_mode_shape_dofs == jj)
            nodes_new[tmp_id] += MODESCALEFCT * modes_mean[ii,jj]
            # tmp_MEAN_MODE[tmp_id] = modes_mean[ii,jj]
        tmp_all_modes = Frequency_dataset[ii][:,8:]
        # plot confidence error ellipses for each floor
        fig_ell, axs_ell = plt.subplots(4,2,figsize=(9,9),facecolor='white')
        fig_ell.suptitle(f'Mode shape at {np.mean(Frequency_dataset[ii][:,0]):.3f} Hz floor by floor', fontweight='bold', fontsize=16)
        fig_ell.legend([Patch(fc='#9BD0D1')], [f'{MODESTDFCT_ELLIPSES} Std. Dev.'],loc='upper center',bbox_to_anchor=(0.5,0.95))
        indexes=np.array([[pp, pp+3] for pp in range(0,12,3)])
        tmp_pearson_array = np.zeros((4,2,3))
        for jj, ind in enumerate(indexes):
            axs_ell[jj,0].plot(triangle(_nodes[ind[0]:ind[1],0]), triangle(_nodes[ind[0]:ind[1],1]), 'o--', color='gray', zorder=1)
            axs_ell[jj,0].plot(triangle(nodes_new[ind[0]:ind[1],0]), triangle(nodes_new[ind[0]:ind[1],1]), 'o-', color='#023e7d', zorder=1)

            axs_ell[jj,1].plot(triangle(_nodes[ind[0]+12:ind[1]+12,0]), triangle(_nodes[ind[0]+12:ind[1]+12,1]), 'o--', color='gray', zorder=1)
            axs_ell[jj,1].plot(triangle(nodes_new[ind[0]+12:ind[1]+12,0]), triangle(nodes_new[ind[0]+12:ind[1]+12,1]), 'o-', color='#023e7d', zorder=1)

            Alignment_color = ['blue','red','#008000']
            for kk in range(3): # cycle for the three monitored points of the floor
                axs_ell[jj,0].scatter(_nodes[ind[0]+kk,0], _nodes[ind[0]+kk,1], color=Alignment_color[kk], marker='o', zorder=2)
                axs_ell[jj,0].scatter(nodes_new[ind[0]+kk,0], nodes_new[ind[0]+kk,1], color=Alignment_color[kk], marker='o', zorder=2)
                pearson = confidence_ellipse(tmp_all_modes[:,connectivity_mode_shape_dofs[ind[0]+kk,0]], tmp_all_modes[:,connectivity_mode_shape_dofs[ind[0]+kk,1]], \
                                axs_ell[jj,0], facecolor='#9BD0D1', n_std=MODESTDFCT_ELLIPSES, center=[nodes_new[ind[0]+kk,0], nodes_new[ind[0]+kk,1]], alpha=0.9) 
                tmp_pearson_array[jj,0,kk] = pearson

                axs_ell[jj,1].scatter(_nodes[ind[0]+12+kk,0], _nodes[ind[0]+12+kk,1], color=Alignment_color[kk], marker='o', zorder=2)
                axs_ell[jj,1].scatter(nodes_new[ind[0]+12+kk,0], nodes_new[ind[0]+12+kk,1], color=Alignment_color[kk], marker='o', zorder=2)
                pearson = confidence_ellipse(tmp_all_modes[:,connectivity_mode_shape_dofs[ind[0]+12+kk,0]], tmp_all_modes[:,connectivity_mode_shape_dofs[ind[0]+12+kk,1]], \
                                axs_ell[jj,1], facecolor='#9BD0D1', n_std=MODESTDFCT_ELLIPSES, center=[nodes_new[ind[0]+12+kk,0], nodes_new[ind[0]+12+kk,1]], alpha=0.9) 
                tmp_pearson_array[jj,1,kk] = pearson

            axs_ell[jj,0].set_xlabel('x [m]')
            axs_ell[jj,0].set_ylabel('y [m]')
            axs_ell[jj,0].text(0.15, 0.9, f"z={_nodes[ind[0],2]:.2f}m",fontsize=10, ha='center', va='center', transform=axs_ell[jj,0].transAxes)
            # axs_ell[jj,0].autoscale(tight=True)

            axs_ell[jj,1].set_xlabel('x [m]')
            axs_ell[jj,1].set_ylabel('y [m]')
            axs_ell[jj,1].text(0.15, 0.9, f"z={_nodes[ind[0]+12,2]:.2f}m",fontsize=10, ha='center', va='center', transform=axs_ell[jj,1].transAxes)
            # axs_ell[jj,1].autoscale(tight=True)

        for axs_ell in fig_ell.get_axes():
            axs_ell.label_outer()
        Pearson_coeff.append(tmp_pearson_array)
        plt.savefig(RESULTS_PATH + f'/Mode_shape_ellipses_{ii+1}.png', format='png', dpi=300)
        plt.savefig(RESULTS_PATH + f'/Mode_shape_ellipses_{ii+1}.pdf')
        pickle.dump(fig_ell, open(RESULTS_PATH + f'/Mode_shape_ellipses_{ii+1}.pkl','wb'))
        plt.close()
    with open(RESULTS_PATH+'/Pearson_coefficients.npy', 'wb') as f:
        np.save(f, Pearson_coeff)
    return Pearson_coeff


def triangle(a):
    return np.hstack((a,np.array(a[0])))

def plot_modes_for_3d_for_paper(modes_mean, modes_std, freq_mean, _nodes, _connectivity, connectivity_mode_shape_dofs, \
                                MODESCALEFCT, MODESTDFCT, RESULTS_PATH):
    sensors_loc = import_data('DATA/Sensors_locations_tower.txt')
    for ii in range(modes_mean.shape[0]):
        fig,ax = plotModeShape3D(_nodes, _connectivity, kwargs_plot_lines={'color':'gray', 'linestyle':'dashed'})
        ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z')

        ax.scatter(_nodes[np.where(sensors_loc == 'A1')[0],0], _nodes[np.where(sensors_loc == 'A1')[0],1],\
                   _nodes[np.where(sensors_loc == 'A1')[0],2], color='blue', marker='o')
        ax.scatter(_nodes[np.where(sensors_loc == 'A2')[0],0], _nodes[np.where(sensors_loc == 'A2')[0],1], \
                   _nodes[np.where(sensors_loc == 'A2')[0],2], color='red', marker='o')
        ax.scatter(_nodes[np.where(sensors_loc == 'A3')[0],0], _nodes[np.where(sensors_loc == 'A3')[0],1], \
                   _nodes[np.where(sensors_loc == 'A3')[0],2], color='#008000', marker='o')

        nodes_new = np.copy(_nodes)
        nodes_new_infStd = np.zeros_like(_nodes)
        nodes_new_supStd = np.zeros_like(_nodes)
        tmp_MEAN_MODE = np.zeros_like(_nodes)
        tmp_STD_DEV = np.zeros_like(_nodes)
        for jj in range(modes_mean.shape[1]): # cycle on mode shape vector columns and associate mode shape to nodes
            tmp_id = np.where(connectivity_mode_shape_dofs == jj)
            nodes_new[tmp_id] += MODESCALEFCT * modes_mean[ii,jj]
            tmp_MEAN_MODE[tmp_id] = modes_mean[ii,jj]
            nodes_new_infStd[tmp_id] = tmp_MEAN_MODE[tmp_id]
            nodes_new_supStd[tmp_id] = tmp_MEAN_MODE[tmp_id]
            nodes_new_infStd[tmp_id] -= MODESTDFCT * modes_std[ii,jj]
            nodes_new_supStd[tmp_id] += MODESTDFCT * modes_std[ii,jj]
            tmp_STD_DEV[tmp_id] = modes_std[ii,jj]
        fig,ax = plotModeShape3D(nodes_new, _connectivity, hold_on=True, kwargs_plot_lines={'color':'#023e7d', 'linestyle':'solid'}, kwargs_plot_markers={'color':'#023e7d', 'marker':'o'})
        
        ax.scatter(nodes_new[np.where(sensors_loc == 'A1')[0],0], nodes_new[np.where(sensors_loc == 'A1')[0],1], \
                   nodes_new[np.where(sensors_loc == 'A1')[0],2], color='blue', marker='o')
        ax.scatter(nodes_new[np.where(sensors_loc == 'A2')[0],0], nodes_new[np.where(sensors_loc == 'A2')[0],1], \
                   nodes_new[np.where(sensors_loc == 'A2')[0],2], color='red', marker='o')
        ax.scatter(nodes_new[np.where(sensors_loc == 'A3')[0],0], nodes_new[np.where(sensors_loc == 'A3')[0],1], \
                   nodes_new[np.where(sensors_loc == 'A3')[0],2], color='#008000', marker='o') 
        
        plt.title(f'Mode shape at {freq_mean[ii]:.3f} Hz', fontweight='bold')
        custom_lines = [
                    Line2D([0], [0], color='blue', marker='o'),
                    Line2D([0], [0], color='red', marker='o'),
                    Line2D([0], [0], color='#008000', marker='o'),
                    Line2D([0], [0], color='gray',linestyle='dashed', lw=4),
                    Line2D([0], [0], color='#023e7d', lw=4),
                    ]
        plt.legend(custom_lines,['A1','A2','A3','Undeformed','Mode shape'],fontsize=10, loc='upper left')
        ax.set_xlabel('x [m]',labelpad=10); ax.tick_params(axis='x', pad=-8)
        ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=50,va='top',ha='right')
        ax.set_ylabel('y [m]',labelpad=18); ax.tick_params(axis='y', pad=-3)
        ax.set_yticklabels(labels=ax.get_yticklabels(),rotation=-20,va='center_baseline',ha='left')
        ax.set_zlabel('z [m]',labelpad=60); ax.tick_params(axis='z', pad=30)
        # z = ax.get_zlim3d(); # ax.set_zlim3d(0, z[1])
        ax.set_box_aspect([1,1,3]); ax.grid(False)
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)); ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        fig.tight_layout()
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1.2)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.png', format='png', dpi=300,bbox_inches='tight')
        plt.savefig(RESULTS_PATH + f'/Mode_shape_{ii+1}.pdf', bbox_inches='tight')
        pickle.dump(fig, open(RESULTS_PATH + f'/Mode_shape_{ii+1}.pkl','wb'))
        plt.close()
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}.txt', nodes_new, fmt='%.4f') 
        np.savetxt(RESULTS_PATH + f'/Mode_shape_{ii+1}_STD_DEV.txt', tmp_STD_DEV) 

        # plot verticalized normalized mode shapes for every alignment with the associated uncertainty
        figs_vert, axs_vert = plt.subplots(1,6,figsize=(9,4),facecolor='white')
        axs_vert[0].set_ylabel('z [mm]')
        Alignment_color = ['blue','red','#008000']
        for jj, Alignment in enumerate(['A1','A2','A3']):
            axs_vert[2*jj].plot(tmp_MEAN_MODE[np.where(sensors_loc == Alignment)[0],0], nodes_new[np.where(sensors_loc == Alignment)[0],2], 'o--', c=Alignment_color[jj])
            axs_vert[2*jj].fill_betweenx(nodes_new[np.where(sensors_loc == Alignment)[0],2], \
                                    nodes_new_infStd[np.where(sensors_loc == Alignment)[0],0], nodes_new_supStd[np.where(sensors_loc == Alignment)[0],0], \
                                    alpha=0.8, color='#9BD0D1', zorder=1)
            axs_vert[2*jj+1].plot(tmp_MEAN_MODE[np.where(sensors_loc == Alignment)[0],1], nodes_new[np.where(sensors_loc == Alignment)[0],2], 'o-', c=Alignment_color[jj])
            axs_vert[2*jj+1].fill_betweenx(nodes_new[np.where(sensors_loc == Alignment)[0],2], \
                                    nodes_new_infStd[np.where(sensors_loc == Alignment)[0],1], nodes_new_supStd[np.where(sensors_loc == Alignment)[0],1], \
                                    alpha=0.8, color='#9BD0D1', zorder=1)
        for ax__ in figs_vert.get_axes():
            ax__.label_outer()
        custom_lines = [
                    Line2D([0], [0], color='blue', linestyle='dashed'),
                    Line2D([0], [0], color='blue', linestyle='solid'),
                    Line2D([0], [0], color='red', linestyle='dashed'),
                    Line2D([0], [0], color='red', linestyle='solid'),
                    Line2D([0], [0], color='#008000', linestyle='dashed'),
                    Line2D([0], [0], color='#008000', linestyle='solid'),
                    Patch(facecolor='#9BD0D1', edgecolor=None)
                    ]
        figs_vert.legend(custom_lines,['A1x','A1y','A2x','A2y','A3x','A3y',f'{MODESTDFCT:d} Std. dev.'],fontsize=10, \
                         loc='lower center', bbox_to_anchor=(0.5,0),ncols=7).set_in_layout(False)
        figs_vert.suptitle(f'Normalized mode shape at {freq_mean[ii]:.3f} Hz', fontweight='bold')
        figs_vert.tight_layout()
        figs_vert.subplots_adjust(bottom=0.2, wspace=0.33)
        figs_vert.savefig(RESULTS_PATH + f'/Normalized_mode_shape_{ii+1}.png', format='png', dpi=300)
        figs_vert.savefig(RESULTS_PATH + f'/Normalized_mode_shape_{ii+1}.pdf')
        plt.close()



