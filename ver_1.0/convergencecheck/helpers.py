import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


def plotModeShape2D(_nodes, _connectivity, ax, color='black',style='solid',annotFlag=False):
    num_frames = _connectivity.shape[0]
    for k in range(num_frames):
        x1 = _nodes[_connectivity[k, 0] - 1, 0]
        y1 = _nodes[_connectivity[k, 0] - 1, 1]
        x2 = _nodes[_connectivity[k, 1] - 1, 0]
        y2 = _nodes[_connectivity[k, 1] - 1, 1]
        xx = [x1, x2]; yy = [y1, y2]
        ax.plot(xx, yy, color=color,linestyle=style,zorder=10)
    for i in range(_nodes.shape[0]):
        xs = _nodes[i, 0]; ys = _nodes[i, 1]
        ax.scatter(xs, ys, color=color, marker='o',zorder=10)
        if annotFlag:
            ax.annotate(f'P{i + 1}', xy=(xs, ys), xytext=(3, 3),
                        textcoords='offset points',zorder=10)
            
def plot_modes_for_2d(modes_mean, modes_std, freq_mean, _nodes, _connectivity, MODESCALEFCT, MODESTDFCT, MODE2D_DIRECTION, RESULTS_PATH):
    for ii in range(modes_mean.shape[0]):
        fig,ax = plt.subplots(figsize=(6,5),facecolor='white')
        plotModeShape2D(_nodes, _connectivity, ax, color='black', style='dashed', annotFlag=True)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        nodes_new = np.copy(_nodes)
        if MODE2D_DIRECTION == 'vertical' :
            nodes_new[1:,0] += MODESCALEFCT * modes_mean[ii,:]
            nodes_new_infStd = np.copy(nodes_new)
            nodes_new_supStd = np.copy(nodes_new)
            nodes_new_infStd[1:,0] -= MODESTDFCT * modes_std[ii,:]
            nodes_new_supStd[1:,0] += MODESTDFCT * modes_std[ii,:]
            plotModeShape2D(nodes_new, _connectivity, ax, color='#023e7d', annotFlag=False)
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
            plotModeShape2D(nodes_new, _connectivity, ax, color='#023e7d', annotFlag=False)
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
        plt.close()

def plotModeShape3D(nodes, connectivity, annotatepoints=0, color='black', style='solid', figsize=(5,6), hold_on=False):
    # num_frames = connectivity.shape[0]
    # if hold_on:
    #     _fig=plt.gcf()
    #     ax=plt.gca()
    # else:
    #     _fig = plt.figure(figsize=figsize,facecolor='white')
    #     ax = plt.axes(projection="3d")
    #     setattr(ax, 'annotate3D', drawing_tools_3d.annotate3d)
    # for k in range(num_frames):
    #     x1 = nodes[connectivity[k, 0] - 1, 0]
    #     y1 = nodes[connectivity[k, 0] - 1, 1]
    #     z1 = nodes[connectivity[k, 0] - 1, 2]
    #     x2 = nodes[connectivity[k, 1] - 1, 0]
    #     y2 = nodes[connectivity[k, 1] - 1, 1]
    #     z2 = nodes[connectivity[k, 1] - 1, 2]
    #     xx = [x1, x2]; yy = [y1, y2]; zz = [z1, z2]
    #     ax.plot3D(xx, yy, zz, color=color,linestyle=style)
    # for i in range(nodes.shape[0]):
    #     xs = nodes[i, 0]; ys = nodes[i, 1]; zs = nodes[i, 2]
    #     ax.scatter(xs, ys, zs, color=color, marker='none')
    #     if annotatepoints:
    #         ax.annotate3D(ax,text=f'P{i + 1}', xyz=(xs, ys, zs), xytext=(3, 3), textcoords='offset points')
    # ax.set_xlabel('x [m]',labelpad=10)
    # ax.tick_params(axis='x', pad=-8)
    # ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=50,va='top',ha='right')
    
    # ax.set_ylabel('y [m]',labelpad=18)
    # ax.tick_params(axis='y', pad=-3)
    # ax.set_yticklabels(labels=ax.get_yticklabels(),rotation=-20,va='center_baseline',ha='left')
    
    # ax.set_zlabel('z [m]',labelpad=60)
    # ax.tick_params(axis='z', pad=30)
    # # z = ax.get_zlim3d()
    # # ax.set_zlim3d(0, z[1])
    # ax.set_box_aspect([1,1,3])
    # ax.grid(False)
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # _fig.tight_layout()
    # _fig.subplots_adjust(left=0, right=1, bottom=0, top=1.2)
    # plt.show()
    # return _fig,ax
    pass

def plot_modes_for_3d():
    pass




def plot_trace(modes_trace_covariance, modes_trace_covariance_rel_diff, freq_mean, CONVMCTHRESH, RESULTS_PATH):
    fig,ax=plt.subplots(figsize=(10,4),facecolor='white')
    fig2,ax2=plt.subplots(figsize=(10,4),facecolor='white')
    for jj in range(modes_trace_covariance.shape[1]):
        ax.plot(np.arange(1,modes_trace_covariance.shape[0]+1), modes_trace_covariance[:,jj],'o-',
                 label=f'Mode at {freq_mean[jj]:.2f} Hz', lw=3)
        ax2.plot(np.arange(2,modes_trace_covariance_rel_diff.shape[0]+2), modes_trace_covariance_rel_diff[:,jj],'o-',
                 label=f'Mode at {freq_mean[jj]:.2f} Hz', lw=1)
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(1)); ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d')); ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.legend(loc='upper left')
    # plt.legend(loc='upper right',bbox_to_anchor=(0.,0.,1.,0.9),ncol=4,framealpha=0.9)
    ax.set_title('Trace covariance matrix Convergence',fontweight='bold')
    ax.set_xlabel('Actually conducted simulations')
    fig.tight_layout()

    ax2.plot(np.arange(2,modes_trace_covariance_rel_diff.shape[0]+2), CONVMCTHRESH*np.ones(modes_trace_covariance_rel_diff.shape[0]),
             'b--',label='Band $\pm$2%')
    ax2.plot(np.arange(2,modes_trace_covariance_rel_diff.shape[0]+2), -CONVMCTHRESH*np.ones(modes_trace_covariance_rel_diff.shape[0]),'b--')
    ax2.fill_between(np.arange(2,modes_trace_covariance_rel_diff.shape[0]+2), CONVMCTHRESH*np.ones(modes_trace_covariance_rel_diff.shape[0]),
                     -CONVMCTHRESH*np.ones(modes_trace_covariance_rel_diff.shape[0]), color='yellow', alpha=0.5)
    # plt.fill_betweenx([-0.1,0.1], 500, 550, color='#63C7B2',alpha=0.5)
    # plt.text(380, -0.045, 'Convergence ASCBR', fontsize=12,color='#63C7B2',fontweight='bold')
    ax2.set_ylim(-0.1,0.1)
    ax2.legend(loc='upper left')
    ax2.set_title('Relative difference on trace covariance matrix Convergence',fontweight='bold')
    ax2.set_xlabel('Actually conducted simulations')
    fig2.tight_layout()

    fig.savefig(RESULTS_PATH + f'/Trace_covariance_matrix_Convergence.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Relative_difference_on_trace_covariance_matrix.png', format='png', dpi=300)

    plt.close('all')

def plot_frequency_conv(freq_mean, freq_std, RESULTS_PATH):
    fig,ax=plt.subplots(figsize=(10,4),facecolor='white')
    fig2,ax2=plt.subplots(figsize=(10,4),facecolor='white')
    for jj in range(freq_mean.shape[1]):
        ax.plot(np.arange(1,freq_mean.shape[0]+1), freq_mean[:,jj],
                label=f'Mode at {freq_mean[-1,jj]:.2f} Hz',lw=3) 
        ax2.plot(np.arange(1,freq_std.shape[0]+1), freq_std[:,jj],
                label=f'Mode at {freq_mean[-1,jj]:.2f} Hz',lw=3) 
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(1)); ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d')); ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.set_ylim(0,ax.get_ylim()[1]*1.20)
    ax.legend(loc='upper left')
    ax.set_title('Average frequency Convergence',fontweight='bold')
    ax.set_xlabel('Actually conducted simulations')
    fig.tight_layout()

    ax2.set_ylim(0,ax2.get_ylim()[1]*1.20)
    ax2.legend(loc='upper left')
    ax2.set_title('Std. dev. frequency Convergence',fontweight='bold')
    ax2.set_xlabel('Actually conducted simulations')
    fig2.tight_layout()

    fig.savefig(RESULTS_PATH + f'/Frequency_Convergence.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Frequency_Std_Convergence.png', format='png', dpi=300)
    plt.close('all')

def plot_damp_conv(damp_mean, damp_std, freq_mean, RESULTS_PATH):
    fig,ax=plt.subplots(figsize=(10,4),facecolor='white')
    fig2,ax2=plt.subplots(figsize=(10,4),facecolor='white')
    for jj in range(damp_mean.shape[1]):
        ax.plot(np.arange(1,damp_mean.shape[0]+1), damp_mean[:,jj],
                label=f'Mode at {freq_mean[jj]:.2f} Hz',lw=3) 
        ax2.plot(np.arange(1,damp_std.shape[0]+1), damp_std[:,jj],
                label=f'Mode at {freq_mean[jj]:.2f} Hz',lw=3) 
    # ax.xaxis.set_major_locator(mticker.MultipleLocator(1)); ax2.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d')); ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    ax.set_ylim(0,ax.get_ylim()[1]*1.20)
    ax.legend(loc='upper left')
    ax.set_title('Average damping ratio convergence',fontweight='bold')
    ax.set_xlabel('Actually conducted simulations')
    fig.tight_layout()

    ax2.set_ylim(0,ax2.get_ylim()[1]*1.20)
    ax2.legend(loc='upper left')
    ax2.set_title('Std. dev. damping ratio Convergence',fontweight='bold')
    ax2.set_xlabel('Actually conducted simulations')
    fig2.tight_layout()

    fig.savefig(RESULTS_PATH + f'/Damping_Convergence.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Damping_Std_Convergence.png', format='png', dpi=300)
    plt.close('all')





