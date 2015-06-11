from scipy.ndimage import imread
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from IPython.html.widgets import interact
from IPython.display import display



def examine_cutouts_diagnostic(catalog, annotated_catalog, cluster_directory='./', field_directory='./', plot_alpha=False, invert_color=False, color_by_user=False, plot_points=True):
    from create_catalogs import outlier_clusters_dbscan
    import mpld3
    from ast import literal_eval

    # some generic things you can configure:
    # what columns we want to plot
    columns = ['ZooID', 'object_id', 'stage', 'field_flavor', 'object_flavor',
               'status', 'markers_in_object', 'markers_in_field', 'people_marking_objects',
               'people_marking_field', 'people_looking_at_field',
               'object_size', 'mean_probability', 'x', 'y']
    def examine_entry(i):
        entry = catalog.iloc[i]
        ZooID = entry['ZooID']
        display(entry[columns])


        cluster_path = cluster_directory + entry['object_name']

        cluster = imread(cluster_path) * 1. / 255.
        if invert_color:
            cluster = np.abs(cluster - cluster.max())

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        axs[0].imshow(cluster)
        cat = annotated_catalog.set_index(['ZooID'])

        entry = cat.loc[ZooID]
        # it is possible that we have ones from multiple stages
        if len(entry) == 2:
            # two stages, choose the one with the correct stage
            truth = entry['stage'].values == catalog.iloc[i]['stage']
            ith = int(np.argwhere(truth))
            entry = entry.iloc[ith]  # needs to be an int to return series. if array, returns dataframe


        if plot_points:
            x_unflat = literal_eval(entry['At_X'])
            y_unflat = literal_eval(entry['At_Y'])
            # flatten out x and y. also cut out empty entries
            x = np.array([xi for xj in x_unflat for xi in xj])
            y = np.array([yi for yj in y_unflat for yi in yj])

            users = np.array([xj for xj in xrange(len(x_unflat)) for xi in x_unflat[xj]])
            binusers = np.bincount(users)
            w = 1. / binusers[users]

            cluster_centers, cluster_labels, labels = outlier_clusters_dbscan(x, y)
            if color_by_user:
                c = users
            else:
                c = labels
            points = axs[1].scatter(x, y, c=c, s=50, alpha=0.8, cmap=plt.cm.Accent)
            tooltiplabels = ['({0}, {1}, {2})'.format(labels[i], users[i], w[i]) for i in xrange(len(labels))]
            tooltip = mpld3.plugins.PointLabelTooltip(points, tooltiplabels)
            mpld3.plugins.connect(fig, tooltip)

        field = imread(field_directory + ZooID + '.png') * 1. / 255. #entry['ZooID'] + '.png')
        if plot_alpha:
            if np.shape(field)[-1] == 4:
                field = field[:, :, 3]
        else:
            field = field[:, :, :3]
        if invert_color:
            field = np.abs(field - field.max())
        IM = axs[1].imshow(field)
        mpld3.plugins.connect(fig, mpld3.plugins.MousePosition(fontsize=14))


    interact(examine_entry,
             i=(0, len(catalog)-1))




# for everyone else
def examine_cutouts(catalog, cluster_directory='./', invert_color=False):
    # some generic things you can configure:
    # what columns we want to plot

    columns = ['ZooID', 'object_id', 'stage', 'field_flavor', 'object_flavor',
               'status', 'markers_in_object', 'markers_in_field', 'people_marking_objects',
               'people_marking_field', 'people_looking_at_field',
               'object_size', 'mean_probability', 'x', 'y']
    def examine_entry(i):
        entry = catalog.iloc[i]
        ZooID = entry['ZooID']
        display(entry[columns])


        cluster_path = cluster_directory + entry['object_name']

        cluster = imread(cluster_path) * 1. / 255.
        if invert_color:
            cluster = np.abs(cluster - cluster.max())

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        cluster = imread(cluster_path)
        axs.imshow(cluster)

    interact(examine_entry,
             i=(0, len(catalog)-1),
             show_field=False)







def examine_catalog(catalog):
    def examine_cat(groupby_str):
        groupby_list = eval(groupby_str)
        print(catalog.groupby(groupby_list).apply(len))

    interact(examine_cat,
             groupby_str="['stage', 'field_flavor', 'object_flavor']")
