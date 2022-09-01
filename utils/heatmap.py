from seaborn import heatmap


def heatmap_cca_cka_sims(exp_date,
                        cca_sims_reg, 
                        cka_sims_reg, 
                        data_set_name, sim_between, n_test_samples,
                        cca_n_components='max', specified_ind=-1,
                        regular_test_acc=None,
                        show_res_fig=False, save_res_fig=True, path='plots_heatmaps'):
    '''
    For sim_between = 'every_layer':
    Similarities are expected to be 2D numpy arrays
    '''

    # sim_between = 'two_nets' 'every_layer'
    cmap ='magma'

    col_if_iid_reg = 'k'

    fig, axs_sim = plt.subplots(2, 1, figsize=(24, 22))
    sns.heatmap(cca_sims_reg,
                vmin=0.0, vmax=1.0, ax=axs_sim[0], cmap=cmap, xticklabels=2, yticklabels=2,
                cbar=True, cbar_kws={'shrink': 0.9}, square=True)
    axs_sim[0].set_ylabel('Layer', size='x-large')
    axs_sim[0].set_xlabel('Layer', size='x-large')
    if regular_test_acc is not None:
        regular_test_acc = '%0.2f' % (100 * regular_test_acc) + ' %'
        axs_sim[0].set_title(f'Regular test set;\n\naccuracy: {regular_test_acc}', size='xx-large', c=col_if_iid_reg)
    else:
        axs_sim[0].set_title('Regular test set\n', size='xx-large', c=col_if_iid_reg)

    sns.heatmap(cka_sims_reg,
                vmin=0.0, vmax=1.0, ax=axs_sim[1], cmap=cmap, xticklabels=2, yticklabels=2,
                cbar=True, square=True, cbar_kws={'shrink': 0.9})  # size='large'
    axs_sim[1].set_xlabel('Layer', size='x-large')
    axs_sim[1].set_ylabel('Layer', size='x-large')

    common_title = titles_dict[sim_between]
    plt.suptitle(common_title, size='xx-large')  # , y=0.98
    # fig.tight_layout()

    if not show_res_fig:
        plt.close(fig)

    if save_res_fig:
        name_conf = os.path.join(path, exp_date + f'_heatmap_{n_test_samples}_sims_{sim_between}')
        fig.savefig(name_conf + '.jpeg')