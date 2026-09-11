import rasterio, numpy as np, time, os
from rasterio.windows import from_bounds
os.environ['GDAL_HTTP_MAX_RETRY']='5'; os.environ['GDAL_HTTP_RETRY_DELAY']='10'; os.environ['GDAL_DISABLE_READDIR_ON_OPEN']='EMPTY_DIR'
files={'soc':"https://zenodo.org/api/records/2525553/files/sol_organic.carbon_usda.6a1c_m_250m_b0..0cm_1950..2017_v0.2.tif/content",
       'clay':"https://zenodo.org/api/records/2525663/files/sol_clay.wfraction_usda.3a1a1a_m_250m_b0..0cm_1950..2017_v0.2.tif/content",
       'ph':"https://zenodo.org/api/records/2525664/files/sol_ph.h2o_usda.4c1a2a_m_250m_b0..0cm_1950..2017_v0.2.tif/content"}
for k,u in files.items():
    t0=time.time()
    try:
        with rasterio.open('/vsicurl/'+u) as src:
            print(k,src.crs,src.width,src.height,src.dtypes,src.nodata,src.is_tiled,src.block_shapes[0],flush=True)
            w=from_bounds(119.3,21.8,122.1,25.4,src.transform)
            arr=src.read(1,window=w); tr=src.window_transform(w)
            prof=src.profile; prof.update(width=arr.shape[1],height=arr.shape[0],transform=tr,compress='deflate',tiled=False)
            with rasterio.open(f'soil_{k}_taiwan_250m.tif','w',**prof) as dst: dst.write(arr,1)
        v=arr[arr!=src.nodata] if src.nodata is not None else arr
        print(k,'done',arr.shape,round(time.time()-t0),'s','min/median/max',v.min(),np.median(v),v.max(),flush=True)
    except Exception as e:
        print(k,'ERR',repr(e)[:300],flush=True)
