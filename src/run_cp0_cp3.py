"""U14: reproducible traffic/weather/calendar exploration. Run from any directory."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json, hashlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score, adjusted_rand_score, mean_absolute_error
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[11,23,37,51,71]
FOLDERS=['CP0','CP1','CP2','CP3']
def save(df,folder,name):
    p=ROOT/folder/name;p.parent.mkdir(exist_ok=True,parents=True);df.to_csv(p,index=False,float_format='%.7g')
def figsave(folder,name):
    plt.savefig(ROOT/folder/name,dpi=180,bbox_inches='tight',facecolor='white');plt.close()
def main():
    threadpool_limits(limits=2)
    for f in FOLDERS:(ROOT/f).mkdir(exist_ok=True)
    print('Loading and aligning',flush=True)
    raw=pd.read_csv(ROOT/'data/METR-LA.csv',index_col=0,parse_dates=True)
    assert raw.index.is_unique and raw.index.to_series().diff().dropna().eq(pd.Timedelta('5min')).all()
    raw=raw.astype('float32'); dates=pd.DatetimeIndex(sorted(raw.index.normalize().unique()))
    # A naive CSV does not establish its timezone. Main hypothesis: local wall clock.
    # The synthetic 288-slot day on DST change is excluded instead of fabricated.
    dst=pd.Timestamp('2012-03-11'); train_end=dates[int(len(dates)*.70)-1];val_end=dates[int(len(dates)*.85)-1]
    clean=raw.mask(raw<=0)
    missing=clean.loc[:str(train_end.date())].isna().mean();kept=missing[missing<=.10].index
    save(pd.DataFrame({'sensor_id':raw.columns,'missing_train_fraction':missing.values,'retained':raw.columns.isin(kept)}),'CP1','calitate_senzori.csv')
    clean=clean[kept]; quality=clean.isna().groupby(clean.index.normalize()).mean()
    # Training-only sensor x clock-slot median; no interpolation across split boundaries.
    slot=clean.index.hour*12+clean.index.minute//5
    med=clean.loc[:str(train_end.date())].groupby(slot[:len(clean.loc[:str(train_end.date())])]).median()
    med=med.fillna(clean.loc[:str(train_end.date())].median())
    filled=clean.fillna(pd.DataFrame(med.reindex(slot).values,index=clean.index,columns=kept))
    w=pd.read_csv(ROOT/'data/open-meteo-34.06N118.24W91m.csv',skiprows=3)
    original_weather_columns=w.columns.tolist()
    w.columns=['time','temp','apparent','humidity','dew','code','wind','direction','gust','rain','cloud']
    w.index=pd.to_datetime(w.pop('time'),utc=True)
    assert w.index.is_unique and not w.isna().any().any()
    wt=w.copy(); wt.index=wt.index.tz_convert('America/Los_Angeles').tz_localize(None)
    # Exact previous whole-hour key, as requested. No hourly interpolation.
    aligned=wt.reindex(raw.index.floor('h'));aligned.index=raw.index
    aligned.index.name='time_local'
    save(aligned.reset_index(),'CP1','meteo_aliniat_5min.csv')
    aligned_hours=wt.reindex(pd.date_range(raw.index.min().floor('h'),raw.index.max().floor('h'),freq='h'))
    wd=aligned_hours.groupby(aligned_hours.index.normalize()).agg(
        met_temp_mean=('temp','mean'),met_temp_min=('temp','min'),met_temp_max=('temp','max'),
        met_humidity=('humidity','mean'),met_wind=('wind','mean'),met_gust=('gust','max'),
        met_rain_mm=('rain','sum'),met_rain_hours=('rain',lambda s:(s>0).sum()),met_cloud=('cloud','mean'),
        met_dew=('dew','mean'))
    weather_counts=aligned_hours['temp'].groupby(aligned_hours.index.normalize()).count()
    wd.loc[weather_counts<24,:]=np.nan
    # Validate calendar fields against actual dates; use explicit holiday dates table.
    cal_path=ROOT/'data/American Holidays 1722 - 2122'
    holidays=pd.read_csv(cal_path/'US Federal Pay and Leave Holidays 2004 to 2100.csv')
    h_actual=pd.to_datetime(holidays['Date']);h_observed=pd.to_datetime(holidays.iloc[:,3])
    calendar_raw=pd.read_csv(cal_path/'400_Years_of_Generated_Dates_and_Holidays.csv',low_memory=False)
    calendar_raw['_date']=pd.to_datetime(calendar_raw['A_DATE'],format='%m/%d/%Y')
    cal2012=calendar_raw[calendar_raw['_date'].isin(dates)].copy()
    save(cal2012,'CP1','calendar_sursa_2012.csv')
    cal=pd.DataFrame(index=dates)
    cal['cal_dow_sin']=np.sin(2*np.pi*dates.dayofweek/7);cal['cal_dow_cos']=np.cos(2*np.pi*dates.dayofweek/7)
    cal['cal_weekend']=(dates.dayofweek>=5).astype(int);cal['cal_holiday']=dates.isin(pd.DatetimeIndex(h_actual).union(pd.DatetimeIndex(h_observed))).astype(int)
    # Daily profiles: 24 means from the 288 raw slots, plus interpretable statistics.
    hour=filled.groupby(filled.index.floor('h')).mean()
    rows=[]
    for sensor in kept:
        profile=hour[sensor].groupby([hour.index.normalize(),hour.index.hour]).mean().unstack()
        profile.columns=[f'trf_h{h:02d}' for h in profile.columns]
        g=filled[sensor].groupby(filled.index.normalize())
        profile['trf_mean']=g.mean();profile['trf_std']=g.std();profile['trf_min']=g.min()
        profile['trf_congested_fraction']=filled[sensor].lt(40).groupby(filled.index.normalize()).mean()
        profile['missing_fraction']=quality[sensor]
        profile['sensor_id']=str(sensor);profile['date']=profile.index
        rows.append(profile.join(wd).join(cal).reset_index(drop=True))
    df=pd.concat(rows,ignore_index=True).sort_values(['date','sensor_id']).reset_index(drop=True)
    valid=(df['missing_fraction']<=.20)&df['met_temp_mean'].notna()&df['date'].ne(dst)
    excluded=df.loc[~valid,['sensor_id','date','missing_fraction','met_temp_mean']]
    save(excluded,'CP1','randuri_excluse.csv');df=df.loc[valid].reset_index(drop=True)
    df['split']=np.where(df.date<=train_end,'train',np.where(df.date<=val_end,'validation','test'))
    # No future outcome used as a feature. Next-day outcome must stay inside same split.
    lookup=df.set_index(['sensor_id','date'])
    nextkeys=pd.MultiIndex.from_arrays([df.sensor_id,df.date+pd.Timedelta(days=1)],names=['sensor_id','date'])
    df['target_nextday_mean']=lookup.trf_mean.reindex(nextkeys).values
    nextsplit=lookup['split'].reindex(nextkeys).values
    df.loc[nextsplit!=df['split'],'target_nextday_mean']=np.nan
    save(df,'CP2','matrice_X_U14.csv')
    save(df[['sensor_id','date','missing_fraction','split','met_rain_mm','cal_holiday']],'CP1','tabel_aliniat_zilnic.csv')
    # Compact wide storage preserves every 5-minute measurement and missingness.
    np.savez_compressed(ROOT/'CP1_v2/trafic_aliniat_5min.npz',time=raw.index.astype(str).values.astype('U19'),sensor=kept.values.astype('U'),speed_mph=clean.values)
    sample=clean.iloc[:24,:3].stack(dropna=False).rename('speed_mph').reset_index()
    sample.columns=['time_local','sensor_id','speed_mph'];sample=sample.merge(aligned.reset_index(),on='time_local',how='left')
    save(sample,'CP1','exemplu_join_5min.csv')
    groups={'T':[c for c in df if c.startswith('trf_')], 'M':[c for c in df if c.startswith('met_')], 'C':[c for c in df if c.startswith('cal_')]}
    feature_rows=[]
    for group,cols in groups.items():
        for c in cols:
            unit='mph' if c.startswith('trf_') else 'unitless'
            if 'fraction' in c:unit='fraction 0..1'
            if c.startswith(('met_temp','met_dew')):unit='degC'
            if c in ['met_wind','met_gust']:unit='km/h'
            if c in ['met_humidity','met_cloud']:unit='percent'
            if c=='met_rain_mm':unit='mm (sum of unique hourly labels)'
            if c=='met_rain_hours':unit='hour labels with rain>0'
            feature_rows.append({'feature':c,'source':group,'unit':unit,'aggregation':'hourly mean from 12 traffic slots' if c.startswith('trf_h') else 'daily summary; see datacard','used_in_model':bool(df.loc[df.split.eq('train'),c].std()>1e-9)})
    save(pd.DataFrame(feature_rows),'CP2','dictionar_trasaturi.csv')
    save(df[sum(groups.values(),[])].describe().T.reset_index(names='feature'),'CP2','statistici_descriptive.csv')
    cross=df.groupby('date')[['trf_mean','trf_congested_fraction']+groups['M']+groups['C']].mean()
    corr=cross.corr(method='spearman');save(corr.reset_index(names='feature'),'CP3','corelatii_spearman_pe_zile.csv')
    plt.figure(figsize=(10,5));plt.imshow(corr.iloc[:2,2:],vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');plt.xticks(range(len(corr.columns)-2),corr.columns[2:],rotation=60,ha='right',fontsize=8);plt.yticks([0,1],['Viteza medie','Fracție <40 mph']);plt.colorbar(label='Spearman ρ; o observație = o zi');plt.tight_layout();figsave('CP3','harta_relatii.png')
    # Timezone sensitivity: alternate hypothesis traffic timestamps are UTC.
    utc_daily=w.reindex(pd.date_range(raw.index.min(),raw.index.max(),freq='h',tz='UTC')).resample('D').agg({'temp':'mean','rain':'sum'})
    utc_daily.index=utc_daily.index.tz_localize(None)
    sens=pd.DataFrame({'date':wd.index,'rain_local':wd.met_rain_mm,'rain_utc_hypothesis':utc_daily.rain.reindex(wd.index)})
    save(sens,'CP3','sensibilitate_fus_orar.csv')
    masks={s:df.split.eq(s).values for s in ['train','validation','test']};tr=masks['train'];va=masks['validation'];te=masks['test']
    blocks={};scalers={}
    for name,cols in groups.items():
        cols=[c for c in cols if df.loc[tr,c].std()>1e-9]
        groups[name]=cols
        imp=SimpleImputer(strategy='median');v=imp.fit(df.loc[tr,cols]).transform(df[cols]);sc=StandardScaler().fit(v[tr]);blocks[name]=sc.transform(v)/np.sqrt(len(cols));scalers[name]=sc
    views={'T':['T'],'M':['M'],'C':['C'],'TM':['T','M'],'TC':['T','C'],'TMC':['T','M','C']}
    results=[];points=[];loadings=[];boot=[];profiles=[];scree=[]
    rng=np.random.default_rng(2026)
    eval_idx=rng.choice(np.where(te)[0],size=min(700,te.sum()),replace=False)
    for view,parts in views.items():
        A=np.concatenate([blocks[p] for p in parts],axis=1);cols=sum([groups[p] for p in parts],[])
        maxk=min(10,A.shape[1]);pca=PCA(n_components=maxk,svd_solver='full').fit(A[tr]);P=pca.transform(A)
        for j,ratio in enumerate(pca.explained_variance_ratio_):scree.append({'view':view,'component':j+1,'explained_variance_ratio':ratio})
        for j in range(min(5,maxk)):
            for c,value in zip(cols,pca.components_[j]):loadings.append({'view':view,'component':j+1,'feature':c,'loading':value})
        print('View',view,A.shape,flush=True)
        for seed in SEEDS:
            # Nystrom kernel map approximates RBF kernel PCA and supports unseen days.
            ny=Nystroem(kernel='rbf',gamma=1/len(parts),n_components=96,random_state=seed)
            B=ny.fit(A[tr]).transform(A);kp=PCA(n_components=maxk,svd_solver='full').fit(B[tr]);K=kp.transform(B)
            for method,Z in [('PCA',P),('RBF-Nystrom-PCA',K)]:
                for k in sorted(set([2,min(5,maxk),maxk])):
                    z=Z[:,:k]
                    inv=Ridge(alpha=1).fit(z[tr],A[tr])
                    rec=pca.inverse_transform(np.pad(z,((0,0),(0,maxk-k)))) if method=='PCA' else inv.predict(z)
                    trust=trustworthiness(A[eval_idx],z[eval_idx],n_neighbors=10)
                    stress_rho=float(spearmanr(pdist(A[eval_idx]),pdist(z[eval_idx])).statistic)
                    y=df.target_nextday_mean.values;use=np.isfinite(y)
                    down=Ridge(alpha=10).fit(z[tr&use],y[tr&use]);pred=down.predict(z)
                    errors=pd.DataFrame({'date':df.loc[te&use,'date'].values,'error':np.abs(pred[te&use]-y[te&use])}).groupby('date').error.mean()
                    # Paired day-level error bootstrap is exported for the reference seed.
                    if seed==SEEDS[0]:save(errors.reset_index().assign(view=view,method=method,k=k),'CP3',f'erori_zilnice_{view}_{method}_{k}.csv')
                    results.append({'view':view,'method':method,'k':k,'seed':seed,'trustworthiness':trust,'distance_spearman':stress_rho,'reconstruction_mse':float(np.mean((rec[te]-A[te])**2)),'nextday_mae_mph':float(errors.mean())})
        if view=='TMC':
            plt.figure(figsize=(7,4));plt.plot(np.arange(1,maxk+1),np.cumsum(pca.explained_variance_ratio_),'o-');plt.axhline(.8,color='gray',ls='--');plt.xlabel('Componente PCA');plt.ylabel('Varianță cumulată (train)');plt.ylim(0,1.03);plt.grid(alpha=.2);figsave('CP2','scree_PCA.png')
    save(pd.DataFrame(results),'CP3','metrici_toate_rularile.csv')
    save(pd.DataFrame(loadings),'CP2','PCA_loadings.csv')
    save(pd.DataFrame(scree),'CP2','PCA_varianta.csv')
    print('CP0-CP3 completed')
if __name__=='__main__':main()
