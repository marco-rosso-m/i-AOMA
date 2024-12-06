import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter




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
    fig.savefig(RESULTS_PATH + f'/Trace_covariance_matrix_Convergence.pdf')
    fig2.savefig(RESULTS_PATH + f'/Relative_difference_on_trace_covariance_matrix.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Relative_difference_on_trace_covariance_matrix.pdf')

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
    fig.savefig(RESULTS_PATH + f'/Frequency_Convergence.pdf')
    fig2.savefig(RESULTS_PATH + f'/Frequency_Std_Convergence.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Frequency_Std_Convergence.pdf')
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
    fig.savefig(RESULTS_PATH + f'/Damping_Convergence.pdf')
    fig2.savefig(RESULTS_PATH + f'/Damping_Std_Convergence.png', format='png', dpi=300)
    fig2.savefig(RESULTS_PATH + f'/Damping_Std_Convergence.pdf')
    plt.close('all')
