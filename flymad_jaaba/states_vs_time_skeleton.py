tempdf = pd.DataFrame()
for fn in glob.glob(_directory ):
    p = pd.read_pickle(fn)
    GROUP = p.ix[0]['GROUP']
    FLY_ID = p.ix[0]['FLY_ID']
    df = p.resample('1s')
    df['GROUP'] = GROUP
    df['FLY_ID'] = FLY_ID
    df['DATE'] = FLY_ID.split('_')[1]
    tempdf = pd.concat([tempdf, df], axis=0)

g = tempdf.groupby(['GROUP', tempdf.index])
means = g.mean()
means = 1.0 - means
means.unstack(level=0).plot(xlim=(pd.to_datetime(0), pd.to_datetime(720*1e9)))
plt.show()



ptempdf = pd.DataFrame()
for fn in glob.glob('/nearline/dickson/bathd/FlyMAD_data_archive/P1_activation_and_silencing/pool_SS01538/*zoom*/states.pickle' ):
    p = pd.read_pickle(fn)
    GROUP = p.ix[0]['GROUP']
    FLY_ID = p.ix[0]['FLY_ID']
    df = p.resample('1s')
    df['GROUP'] = GROUP
    df['FLY_ID'] = FLY_ID
    ptempdf = pd.concat([ptempdf, df], axis=0)

pg = ptempdf.groupby(['GROUP', ptempdf.index])
pmeans = pg.mean()
pmeans = 1.0 - pmeans
pmeans.unstack(level=0).plot(xlim=(pd.to_datetime(0), pd.to_datetime(600*1e9)))
plt.show()





def plot_data(means, sems, ns, measurement):
    means = means[means[measurement].notnull()]
    #means = 1.0 - means
    #ns = ns[ns[measurement].notnull()]
    #sems = sems[sems[measurement].notnull()]
    fig = plt.figure()
    group_number = 0
    ax = fig.add_subplot(1,1,1)
    y_range = []
    x_range = []
    laser_x = []
    for x in means.index.levels[0]:
        max_n = ns.ix[x]['FLY_ID'].max()
        x_values = []
        y_values = []
        psems = []
        nsems = []
        for w in means.ix[x].index:
            laser_x.append((w-pd.to_datetime(0)).total_seconds())
            if ns.ix[x]['FLY_ID'][w] > ((max_n)-3): #(max_n/3):
                x_range.append((w-pd.to_datetime(0)).total_seconds())
                #print ns.ix[x]['FlyID'][w]
                x_values.append((w-pd.to_datetime(0)).total_seconds())
                y_values.append(means.ix[x,w][measurement])
                psems.append(sems.ix[x,w][measurement])
                nsems.append(-1.0*sems.ix[x,w][measurement])
        #x_values = list((means.ix[x].index - pd.to_datetime(0)).total_seconds())
        #y_values = list(means.ix[x][measurement])
        #psems = list(sems.ix[x][measurement])
        #nsems = list(-1*(sems.ix[x][measurement]))
        y_range.append(np.amin(y_values))
        y_range.append(np.amax(y_values))
        top_errbar = tuple(map(sum, zip(psems, y_values)))
        bottom_errbar = tuple(map(sum, zip(nsems, y_values)))
        p = plt.plot(x_values, y_values, linewidth=3, zorder=100,
                        linestyle = '-',
                        color=colourlist[group_number],
                        label=(x + ', n= ' + str(max_n))) 
        q = plt.fill_between(x_values, 
                            top_errbar, 
                            bottom_errbar, 
                            alpha=0.15, 
                            linewidth=0,
                            zorder=90,
                            color=colourlist[group_number],
                            )
        group_number += 1
    if 'maxWingAngle' in measurement:
        ax.set_ylabel('Mean maximum wing angle (rad)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
    elif 'dtarget' in measurement:
        ax.set_ylabel('Mean min. distance to target (mm)' + ' ' + u"\u00B1" + ' SEM', fontsize=16)   # +/- sign is u"\u00B1"
        
    else:
        ax.set_ylabel('Mean ' + measurement  + ' ' + u"\u00B1" + ' SEM', fontsize=16)
        
    ax.set_xlabel('Time (s)', fontsize=16)      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plt.show()


ddf = pd.DataFrame()
for fn in glob.glob('/tier2/dickson/bathd/FlyMAD/data_for_processing/160809_pooled_data/*zoom*/frame_by_frame_synced.pickle' ):
    p = pd.read_pickle(fn)
    GROUP = p.ix[0]['group']
    FLY_ID = fn.split('/')[-2]#[p.ix[0]['FLY_ID']
    df = pd.DataFrame()
    df = p[['Laser2_state',parameter]]
    df['GROUP'] = GROUP
    df['FLY_ID'] = FLY_ID
    ddf = pd.concat([ddf, df], axis=0)

stims = ['0','200','633','2000','6310','20000']

ddf = pd.DataFrame()
for fn in glob.glob('/tier2/dickson/bathd/FlyMAD/data_for_processing/160809_pooled_data/*zoom*/states.pickle' ):
    p = pd.read_pickle(fn)
    GROUP = p.ix[0]['GROUP']
    FLY_ID = fn.split('/')[-2]#[p.ix[0]['FLY_ID']
    df = pd.DataFrame()
    df = p[['Laser2_state',parameter]]
    df['GROUP'] = GROUP
    df['FLY_ID'] = FLY_ID
    ddf = pd.concat([ddf, df], axis=0)

parameter = 'state'     

fig = plt.figure()
for grp in range(len(set(ddf.GROUP))):
    foo = ddf[ddf['GROUP'] == stims[grp]]
    position = str(len(set(ddf.GROUP)))+ str(1) + str(grp+1)
    plot_trials_and_mean(foo, position, parameter)
plt.show()
       

    
def plot_trials_and_mean(df, fig_position, parameter):
    
    ax = fig.add_subplot(fig_position)
    for fly in list(set(df.FLY_ID)):
        foo = df[df['FLY_ID'] == fly]
        plt.plot(foo.index, foo[parameter], color='#4d4d4d', linewidth=0.5, alpha=0.3, zorder=2)
    g = df.resample('67ms')
    dates = g.index.to_pydatetime()
    x_val = mdates.date2num(dates)
    plt.plot(g.index, g[parameter], color="#FFAA20", linewidth=4, zorder=3)

    laser_2 = collections.BrokenBarHCollection.span_where(x_val, ymin=0.85*(df[parameter].min()), ymax=1.15*(df[parameter].max()), where=g['Laser2_state'] > 0.1, facecolor='#FFB2B2', linewidth=0, edgecolor=None, alpha=1.0, zorder=1) #red FFB2B2
    ax.add_collection(laser_2)
    ax.set_ylim(0.85*(df[parameter].min()),1.15*(df[parameter].max()))
    ax.set_xlim(719162.999,719163.007)
    ax.set_ylabel(parameter, fontsize=16)
    ax.set_xlabel('Time (hh:mm:ss)', fontsize=16)      
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    return

